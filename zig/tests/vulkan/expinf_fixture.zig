//! tf-expinf-probe's host-side references: the f64 expected value per input per formula,
//! the exact class match and the finite tolerance. No device, no Vulkan import — pure
//! host logic, tested below.
const std = @import("std");

pub const formulas = [_][]const u8{
    "exp(x)", "exp(-x)", "exp(-abs(x))", "exp2(x)", "exp(-(1.0*x))",
    "exp(-(exp(z)*x))", "softplus(x)", "exp(-(exp(z)*softplus(x)))",
    "exp(-(exp(z)*softplus(bf16_safe(x))))",
};

pub const n_inputs = 7; // indices 0..6; index 7 of the buffer is the runtime zero z
pub const n_formulas = formulas.len;
pub const outputs = n_inputs * n_formulas;

/// The buffer layout: one uint bit pattern per slot; slot 7 is the runtime zero z.
pub const input_bits = [8]u32{
    0x7f800000, // +inf
    0xff800000, // -inf
    @bitCast(@as(f32, -88.0)),
    @bitCast(@as(f32, -104.0)), // below the f32 exp underflow
    @bitCast(@as(f32, 0.0)),
    @bitCast(@as(f32, 1.0)),
    0x7fc00000, // +NaN
    @bitCast(@as(f32, 0.0)), // z, read by the shader at runtime
};

pub const input_names = [_][]const u8{ "+inf", "-inf", "-88.0", "-104.0", "0.0", "1.0", "+NaN" };

pub const Class = enum { nan, pinf, ninf, zero, finite };

pub fn classify(v: anytype) Class {
    if (std.math.isNan(v)) return .nan;
    if (std.math.isPositiveInf(v)) return .pinf;
    if (std.math.isNegativeInf(v)) return .ninf;
    if (v == 0) return .zero;
    return .finite;
}

pub const class_names = [_][]const u8{ "nan", "+inf", "-inf", "zero", "finite" };

pub fn className(c: Class) []const u8 {
    return class_names[@intFromEnum(c)];
}

/// f64 expected value of formula `f` at input index `i`. Slot 7 (z = 0) is never an input.
pub fn want(i: usize, f: usize) f64 {
    std.debug.assert(i < n_inputs and f < n_formulas);
    const x: f64 = @as(f32, @bitCast(input_bits[i]));
    const z: f64 = @as(f32, @bitCast(input_bits[7]));
    const sp = softplus64(x);
    const spb = softplus64(bf16Safe64(x));
    return switch (f) {
        0 => @exp(x),
        1 => @exp(-x),
        2 => @exp(-@abs(x)),
        3 => @exp2(x),
        4 => @exp(-1.0 * x),
        5 => @exp(-(@exp(z) * x)),
        6 => sp,
        7 => @exp(-(@exp(z) * sp)),
        8 => @exp(-(@exp(z) * spb)),
        else => unreachable,
    };
}

fn softplus64(x: f64) f64 {
    return @max(x, 0.0) + std.math.log1p(@exp(-@abs(x)));
}

/// The fixed bf16 path in f64: RNE to bf16 with a NaN guard, widened back. NaN never
/// escapes; +/-inf and the f32 finite range round-trip through bf16 widening.
fn bf16Safe64(v: f64) f64 {
    const x: f32 = @floatCast(v);
    if (std.math.isNan(x)) return @as(f32, @bitCast(@as(u32, 0x7fc00000)));
    const u: u32 = @bitCast(x);
    const rounded = (u +% 0x7fff +% ((u >> 16) & 1)) & 0xffff0000;
    return @as(f32, @bitCast(rounded));
}

/// The f64 expectation cast to what an IEEE f32 result must be: values above the f32
/// maximum overflow to +inf, tiny values land in the underflow window the caller checks.
pub fn want32(i: usize, f: usize) f32 {
    return @floatCast(want(i, f));
}

/// The exp argument of formula `f` at input `i` in f64 — the driver's exp/exp2 bound is
/// 3 + 2*|arg| ulp (GLSL.std.450), so the tolerance derives from it per formula.
pub fn expArg(i: usize, f: usize) f64 {
    std.debug.assert(i < n_inputs and f < n_formulas);
    const x: f64 = @as(f32, @bitCast(input_bits[i]));
    const z: f64 = @as(f32, @bitCast(input_bits[7]));
    const sp = softplus64(x);
    const spb = softplus64(bf16Safe64(x));
    return switch (f) {
        0 => x,
        1, 2, 4 => -x,
        3 => x * 1.4426950408889634, // exp2(x) as exp(x*log2 e)
        5 => @exp(z) * x,
        6 => std.math.nan(f64), // softplus: not exp-based, flat tolerance below
        7, 8 => @exp(z) * (if (f == 7) sp else spb),
        else => unreachable,
    };
}

pub const Verdict = union(enum) {
    ok,
    class_mismatch: struct { want: Class, got: Class },
    value_mismatch: struct { want: f64, got: f32 },
};

/// Underflow boundary: an f64 expectation below this accepts exact zero or a value
/// itself below it (denormal or flushed-to-zero).
pub const underflow = 1e-37;

/// Flat ulp tolerance for the non-exp formula (softplus: max + log1p legs).
pub const softplus_ulp = 4;

/// Allowed f32 ulp for formula `f`: GLSL.std.450's 3 + 2*|arg| for the exp/exp2-based
/// formulas, a flat 4 for softplus.
pub fn ulps(i: usize, f: usize) u32 {
    if (f == 6) return softplus_ulp;
    return @intFromFloat(3.0 + 2.0 * @abs(expArg(i, f)));
}

pub fn check(got_bits: u32, i: usize, f: usize) Verdict {
    const w = want(i, f);
    const w32 = want32(i, f);
    const got: f32 = @bitCast(got_bits);
    if (std.math.isNan(w)) {
        if (std.math.isNan(got)) return .ok;
        return .{ .class_mismatch = .{ .want = .nan, .got = classify(got) } };
    }
    const wc = classify(w32);
    const gc = classify(got);
    if (@abs(w) < underflow) {
        // exp underflow: exact zero or denormal-flush-to-zero accepted.
        if (got == 0 or @abs(got) < underflow) return .ok;
        return .{ .value_mismatch = .{ .want = w, .got = got } };
    }
    if (wc != gc) return .{ .class_mismatch = .{ .want = wc, .got = gc } };
    if (wc == .zero) return .ok;
    if (!withinUlp(got, w32, ulps(i, f))) return .{ .value_mismatch = .{ .want = w, .got = got } };
    return .ok;
}

fn withinUlp(got: f32, expected: f32, n: u32) bool {
    if (got == expected) return true;
    var cur = expected;
    for (0..n) |_| {
        cur = std.math.nextAfter(f32, cur, got);
        if (got == cur) return true;
    }
    return false;
}

test "the inf expectations match the IEEE/float64 table" {
    // x = +inf: e1=+inf e2=0 e3=0 e4=+inf e5=0 e6=0 e7=+inf e8=0 e9=0
    try std.testing.expectEqual(Class.pinf, classify(want(0, 0)));
    for ([_]usize{ 1, 2, 4, 5, 7, 8 }) |f| try std.testing.expectEqual(@as(f64, 0), want(0, f));
    try std.testing.expectEqual(Class.pinf, classify(want(0, 3)));
    try std.testing.expectEqual(Class.pinf, classify(want(0, 6)));
    // x = -inf: e1=0 e2=+inf e3=0 e4=0 e5=+inf e6=+inf e7=0 e8=1 e9-chain g=0
    for ([_]usize{ 0, 2, 3 }) |f| try std.testing.expectEqual(@as(f64, 0), want(1, f));
    for ([_]usize{ 4, 5 }) |f| try std.testing.expectEqual(Class.pinf, classify(want(1, f)));
    try std.testing.expectEqual(@as(f64, 0), want(1, 6));
    try std.testing.expectEqual(@as(f64, 1), want(1, 7));
    try std.testing.expectEqual(@as(f64, 1), want(1, 8));
}

test "zero and one inputs give finite exact-class values" {
    try std.testing.expectEqual(@as(f64, 1), want(4, 0));
    // x = 0: softplus = log 2, so the full g = exp(-log 2) = 0.5.
    try std.testing.expectEqual(@as(f64, 0.5), want(4, 7));
    try std.testing.expectEqual(@as(f64, @exp(-88.0)), want(2, 0));
}

test "the checker accepts the honest f32 answers and refuses faults" {
    // exp(-inf) observed as +0: ok for e2..e6-style slots of +inf input.
    try std.testing.expectEqual(Verdict.ok, check(0x00000000, 0, 1));
    // exp(-inf) observed as NaN: a value fault under the underflow rule (and a NaN class).
    try std.testing.expectEqual(.value_mismatch, std.meta.activeTag(check(0x7fc00000, 0, 1)));
    // g for a=+inf observed as NaN (the original driver bug): a value fault.
    try std.testing.expectEqual(.value_mismatch, std.meta.activeTag(check(0x7fc00000, 0, 8)));
    // g for a=+inf observed as 0: ok.
    try std.testing.expectEqual(Verdict.ok, check(0x00000000, 0, 8));
    // NaN input: every formula must come out NaN.
    for (0..n_formulas) |f| try std.testing.expectEqual(Verdict.ok, check(0x7fc00000, 6, f));
    // -inf input, e8 = 1: one ulp off is fine, five is not.
    try std.testing.expectEqual(Verdict.ok, check(@bitCast(@as(f32, 1.0)), 1, 7));
    const near: f32 = std.math.nextAfter(f32, 1.0, std.math.inf(f32));
    try std.testing.expectEqual(Verdict.ok, check(@bitCast(near), 1, 7));
    const far: f32 = @bitCast(@as(u32, @as(u32, @bitCast(@as(f32, 1.0))) + 9));
    try std.testing.expectEqual(.value_mismatch, std.meta.activeTag(check(@bitCast(far), 1, 7)));
    // -88 input, e1 = exp(-88): underflow window accepts zero or a denormal-class value.
    try std.testing.expectEqual(Verdict.ok, check(0x00000000, 2, 0));
    try std.testing.expectEqual(.value_mismatch, std.meta.activeTag(check(@bitCast(@as(f32, 1e-30)), 2, 0)));
}

test "the 3 + 2*|arg| exp tolerance admits the observed -88 result and the overflow rule sends -104's exp(-x) to +inf" {
    // x = -88, exp(-x): f64 expectation ~1.6516e38; the device answered ~15 ulp above,
    // inside GLSL.std.450's 3 + 2*88 = 179-ulp bound for exp.
    const w32 = want32(2, 1);
    try std.testing.expectApproxEqRel(@as(f64, 1.6516362549940018e38), w32, 1e-6);
    const got15: u32 = @as(u32, @bitCast(w32)) + 15;
    try std.testing.expectEqual(Verdict.ok, check(got15, 2, 1));
    // 200 ulp off is outside the bound: a real fault.
    try std.testing.expectEqual(.value_mismatch, std.meta.activeTag(check(@as(u32, @bitCast(w32)) + 200, 2, 1)));
    try std.testing.expectEqual(@as(u32, 179), ulps(2, 1));
    // x = -104, exp(-x): the f64 expectation 1.47e45 overflows f32, so the required f32
    // answer is +inf; inf passes, a large finite value is a class fault.
    try std.testing.expect(want(3, 1) > 1e44);
    try std.testing.expectEqual(Class.pinf, classify(want32(3, 1)));
    try std.testing.expectEqual(Verdict.ok, check(0x7f800000, 3, 1));
    try std.testing.expectEqual(.class_mismatch, std.meta.activeTag(check(@bitCast(@as(f32, 3.4e38)), 3, 1)));
}
