using LinearAlgebra: LinearAlgebra as LA
using MatrixAlgebraKit: MatrixAlgebraKit as MAK
using TensorAlgebra.MatrixAlgebra: MatrixAlgebra as MA
using TensorAlgebra: TensorAlgebra as TA
using TupleTools: TupleTools

# This layer is used to define derivative rules (to skip differentiating `setdiff`).
dimnames_setdiff(s1, s2) = setdiff(s1, s2)

Base.:*(a1::AbstractNamedTensor, a2::AbstractNamedTensor) = mul_nameddims(a1, a2)
function mul_nameddims(a1::AbstractNamedTensor, a2::AbstractNamedTensor)
    a_dest, dimnames_dest = TA.contract(
        unnamed(a1), dimnames(a1), unnamed(a2), dimnames(a2)
    )
    return nameddims(a_dest, dimnames_dest)
end

# Left associative fold/reduction.
# Circumvent Base definitions:
# ```julia
# *(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix)
# *(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix)
# ```
# that optimize matrix multiplication sequence.
function Base.:*(
        a1::AbstractNamedTensor, a2::AbstractNamedTensor,
        a3::AbstractNamedTensor, a_rest::AbstractNamedTensor...
    )
    return mul_nameddims(a1, a2, a3, a_rest...)
end
function mul_nameddims(
        a1::AbstractNamedTensor, a2::AbstractNamedTensor,
        a3::AbstractNamedTensor, a_rest::AbstractNamedTensor...
    )
    return *(*(a1, a2), a3, a_rest...)
end

function LA.mul!(
        a_dest::AbstractNamedTensor,
        a1::AbstractNamedTensor, a2::AbstractNamedTensor,
        α::Number, β::Number
    )
    return mul!_nameddims(a_dest, a1, a2, α, β)
end
function mul!_nameddims(
        a_dest::AbstractNamedTensor,
        a1::AbstractNamedTensor, a2::AbstractNamedTensor,
        α::Number, β::Number
    )
    TA.contractadd!(
        unnamed(a_dest), dimnames(a_dest),
        unnamed(a1), dimnames(a1),
        unnamed(a2), dimnames(a2),
        α, β
    )
    return a_dest
end

function LA.mul!(
        a_dest::AbstractNamedTensor,
        a1::AbstractNamedTensor, a2::AbstractNamedTensor
    )
    return mul!_nameddims(a_dest, a1, a2)
end
function mul!_nameddims(
        a_dest::AbstractNamedTensor,
        a1::AbstractNamedTensor, a2::AbstractNamedTensor
    )
    TA.contract!(
        unnamed(a_dest), dimnames(a_dest),
        unnamed(a1), dimnames(a1),
        unnamed(a2), dimnames(a2)
    )
    return a_dest
end

# Locate the named-dimension groups `group1`, `group2` within `a`, returning their two
# positional index groups.
function nameperm(a::AbstractNamedTensor, group1, group2)
    return TA.biperm(dimnames(a), name.(group1), name.(group2))
end

# i, j, k, l = named.((2, 2, 2, 2), ("i", "j", "k", "l"))
# a = randn(i, j, k, l)
# matricize(a, (i, k) => "a", (j, l) => "b")
function TA.matricize(a::AbstractNamedTensor, fusions::Vararg{Pair, 2})
    return matricize_nameddims(a, fusions...)
end
function matricize_nameddims(na::AbstractNamedTensor, fusions::Vararg{Pair, 2})
    group1, group2 = first.(fusions)
    perm_codomain, perm_domain = nameperm(na, group1, group2)
    a_fused = TA.matricizeperm(unnamed(na), perm_codomain, perm_domain)
    return nameddims(a_fused, last.(fusions))
end

function TA.unmatricize(na::AbstractNamedTensor, splitters::Vararg{Pair, 2})
    return unmatricize_nameddims(na, splitters...)
end
function unmatricize_nameddims(na::AbstractNamedTensor, splitters::Vararg{Pair, 2})
    splitters = name.(first.(splitters)) .=> last.(splitters)
    split_namedlengths = last.(splitters)
    splitters_unnamed = map(splitters) do splitter
        fused_name, split_namedlengths = splitter
        fused_dim = findfirst(isequal(fused_name), dimnames(na))
        split_lengths = unnamed.(split_namedlengths)
        return fused_dim => split_lengths
    end
    blocked_axes = last.(TupleTools.sort(splitters_unnamed; by = first))
    a_split = TA.unmatricize(unnamed(na), blocked_axes...)
    names_split = Any[tuple.(dimnames(na))...]
    for splitter in splitters
        fused_name, split_namedlengths = splitter
        fused_dim = findfirst(isequal(fused_name), dimnames(na))
        split_names = name.(split_namedlengths)
        names_split[fused_dim] = split_names
    end
    names_split = reduce((x, y) -> (x..., y...), names_split)
    return nameddims(a_split, names_split)
end

for f in [
        :left_orth, :left_polar, :lq_compact, :lq_full, :qr_compact, :qr_full,
        :right_orth, :right_polar,
    ]
    f_nameddims = Symbol(f, "_nameddims")
    @eval begin
        function MAK.$f(
                a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...
            )
            return $f_nameddims(a, dimnames_codomain, dimnames_domain; kwargs...)
        end
        function $f_nameddims(
                a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...
            )
            codomain = name.(dimnames_codomain)
            domain = name.(dimnames_domain)
            x_unnamed, y_unnamed =
                TA.$f(unnamed(a), dimnames(a), codomain, domain; kwargs...)
            name_x = uniquename(dimnames(a, 1))
            name_y = name_x
            dimnames_x = (codomain..., name_x)
            dimnames_y = (name_y, domain...)
            x = nameddims(x_unnamed, dimnames_x)
            y = nameddims(y_unnamed, dimnames_y)
            return x, y
        end
        function MAK.$f(a::AbstractNamedTensor, dimnames_codomain; kwargs...)
            return $f_nameddims(a, dimnames_codomain; kwargs...)
        end
        function $f_nameddims(a::AbstractNamedTensor, dimnames_codomain; kwargs...)
            codomain = name.(dimnames_codomain)
            domain = dimnames_setdiff(dimnames(a), codomain)
            return MAK.$f(a, codomain, domain; kwargs...)
        end
    end
end

#
# SVD (three-output).
#

for f in [:svd_compact, :svd_full, :svd_trunc]
    f_nameddims = Symbol(f, "_nameddims")
    @eval begin
        function MAK.$f(
                a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...
            )
            return $f_nameddims(a, dimnames_codomain, dimnames_domain; kwargs...)
        end
        function $f_nameddims(
                a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...
            )
            codomain = name.(dimnames_codomain)
            domain = name.(dimnames_domain)
            u_unnamed, s_unnamed, v_unnamed = TA.$f(
                unnamed(a), dimnames(a), codomain, domain; kwargs...
            )
            name_u = uniquename(dimnames(a, 1))
            name_v = uniquename(dimnames(a, 1))
            dimnames_u = (codomain..., name_u)
            dimnames_s = (name_u, name_v)
            dimnames_v = (name_v, domain...)
            u = nameddims(u_unnamed, dimnames_u)
            s = nameddims(s_unnamed, dimnames_s)
            v = nameddims(v_unnamed, dimnames_v)
            return u, s, v
        end
        function MAK.$f(a::AbstractNamedTensor, dimnames_codomain; kwargs...)
            return $f_nameddims(a, dimnames_codomain; kwargs...)
        end
        function $f_nameddims(a::AbstractNamedTensor, dimnames_codomain; kwargs...)
            return MAK.$f(
                a,
                dimnames_codomain,
                dimnames_setdiff(dimnames(a), name.(dimnames_codomain));
                kwargs...
            )
        end
    end
end

#
# Singular values.
#

function MAK.svd_vals(
        a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...
    )
    return svd_vals_nameddims(a, dimnames_codomain, dimnames_domain; kwargs...)
end
function svd_vals_nameddims(
        a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...
    )
    return TA.svd_vals(
        unnamed(a),
        dimnames(a),
        name.(dimnames_codomain),
        name.(dimnames_domain);
        kwargs...
    )
end

function MAK.svd_vals(a::AbstractNamedTensor, dimnames_codomain; kwargs...)
    return svd_vals_nameddims(a, dimnames_codomain; kwargs...)
end
function svd_vals_nameddims(a::AbstractNamedTensor, dimnames_codomain; kwargs...)
    codomain = name.(dimnames_codomain)
    domain = dimnames_setdiff(dimnames(a), codomain)
    return MAK.svd_vals(a, codomain, domain; kwargs...)
end

#
# Eigendecomposition (two-output).
#

for f in [:eigh_full, :eig_full, :eigh_trunc, :eig_trunc]
    f_nameddims = Symbol(f, "_nameddims")
    @eval begin
        function MAK.$f(
                a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...
            )
            return $f_nameddims(a, dimnames_codomain, dimnames_domain; kwargs...)
        end
        function $f_nameddims(
                a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...
            )
            codomain = name.(dimnames_codomain)
            domain = name.(dimnames_domain)
            d_unnamed, v_unnamed = TA.$f(
                unnamed(a), dimnames(a), codomain, domain; kwargs...
            )
            name_d = uniquename(dimnames(a, 1))
            name_d′ = uniquename(name_d)
            name_v = name_d
            dimnames_d = (name_d′, name_d)
            dimnames_v = (domain..., name_v)
            d = nameddims(d_unnamed, dimnames_d)
            v = nameddims(v_unnamed, dimnames_v)
            return d, v
        end
    end
end

#
# Eigenvalues.
#

for f in [:eigh_vals, :eig_vals]
    f_nameddims = Symbol(f, "_nameddims")
    @eval begin
        function MAK.$f(
                a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...
            )
            return $f_nameddims(a, dimnames_codomain, dimnames_domain; kwargs...)
        end
        function $f_nameddims(
                a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...
            )
            codomain = name.(dimnames_codomain)
            domain = name.(dimnames_domain)
            return TA.$f(unnamed(a), dimnames(a), codomain, domain; kwargs...)
        end
    end
end

function MAK.left_null(
        a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...
    )
    return left_null_nameddims(a, dimnames_codomain, dimnames_domain; kwargs...)
end
function left_null_nameddims(
        a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...
    )
    codomain = name.(dimnames_codomain)
    domain = name.(dimnames_domain)
    n_unnamed = TA.left_null(unnamed(a), dimnames(a), codomain, domain; kwargs...)
    name_n = uniquename(dimnames(a, 1))
    dimnames_n = (codomain..., name_n)
    return nameddims(n_unnamed, dimnames_n)
end

function MAK.left_null(a::AbstractNamedTensor, dimnames_codomain; kwargs...)
    return left_null_nameddims(a, dimnames_codomain; kwargs...)
end
function left_null_nameddims(a::AbstractNamedTensor, dimnames_codomain; kwargs...)
    codomain = name.(dimnames_codomain)
    domain = dimnames_setdiff(dimnames(a), codomain)
    return MAK.left_null(a, codomain, domain; kwargs...)
end

function MAK.right_null(
        a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...
    )
    return right_null_nameddims(a, dimnames_codomain, dimnames_domain; kwargs...)
end
function right_null_nameddims(
        a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...
    )
    codomain = name.(dimnames_codomain)
    domain = name.(dimnames_domain)
    n_unnamed = TA.right_null(unnamed(a), dimnames(a), codomain, domain; kwargs...)
    name_n = uniquename(dimnames(a, 1))
    dimnames_n = (name_n, domain...)
    return nameddims(n_unnamed, dimnames_n)
end

function MAK.right_null(a::AbstractNamedTensor, dimnames_codomain; kwargs...)
    return right_null_nameddims(a, dimnames_codomain; kwargs...)
end
function right_null_nameddims(a::AbstractNamedTensor, dimnames_codomain; kwargs...)
    codomain = name.(dimnames_codomain)
    domain = dimnames_setdiff(dimnames(a), codomain)
    return MAK.right_null(a, codomain, domain; kwargs...)
end

"""
    TensorAlgebra.MatrixAlgebra.gram_eigh_full(a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...) -> x

Gram factorization of a Hermitian positive semi-definite named array `a`,
returning `x` such that `a ≈ x * x_cod`, where `x_cod` is `conj(x)` with
its domain dimension names replaced by the corresponding codomain names.
`x` carries the domain dimension names of `a` (matching the convention
that the stored factor labels a vector in `a`'s input space) and a fresh
trailing rank name.

`kwargs` are forwarded to `TensorAlgebra.gram_eigh_full` on the underlying
unnamed array (e.g. `atol`, `rtol`).

# Examples

```jldoctest
julia> using ITensorBase: dimnames, namedoneto, replacedimnames

julia> using TensorAlgebra.MatrixAlgebra: gram_eigh_full

julia> i, j, k, l, aux = namedoneto.((2, 2, 2, 2, 8), ("i", "j", "k", "l", "aux"));

julia> b = randn(aux, i, k);

julia> a = conj(b) * replacedimnames(b, "i" => "j", "k" => "l");

julia> x = gram_eigh_full(a, (i, k), (j, l));

julia> replacedimnames(x, "j" => "i", "l" => "k") * conj(x) ≈ a
true
```
"""
function MA.gram_eigh_full(
        a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...
    )
    return gram_eigh_full_nameddims(a, dimnames_codomain, dimnames_domain; kwargs...)
end
function gram_eigh_full_nameddims(
        a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...
    )
    codomain = name.(dimnames_codomain)
    domain = name.(dimnames_domain)
    x_unnamed = TA.gram_eigh_full(unnamed(a), dimnames(a), codomain, domain; kwargs...)
    name_x = uniquename(dimnames(a, 1))
    dimnames_x = (domain..., name_x)
    return nameddims(x_unnamed, dimnames_x)
end

"""
    TensorAlgebra.MatrixAlgebra.gram_eigh_full_with_pinv(a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...) -> x, y

Like `TensorAlgebra.MatrixAlgebra.gram_eigh_full`, but additionally returns a
named array `y` that is a left inverse of `x`: `y * x ≈ I` on the rank
subspace (equal to the identity when `a` is full rank). `x` has the
rank-name last, `y` has it first, both sharing the domain dimension
names of `a`.

# Examples

```jldoctest
julia> using LinearAlgebra: I

julia> using ITensorBase: unname, dimnames, namedoneto, replacedimnames

julia> using TensorAlgebra.MatrixAlgebra: gram_eigh_full_with_pinv

julia> i, j, k, l, aux = namedoneto.((2, 2, 2, 2, 8), ("i", "j", "k", "l", "aux"));

julia> b = randn(aux, i, k);

julia> a = conj(b) * replacedimnames(b, "i" => "j", "k" => "l");

julia> x, y = gram_eigh_full_with_pinv(a, (i, k), (j, l));

julia> replacedimnames(x, "j" => "i", "l" => "k") * conj(x) ≈ a
true

julia> rname = only(setdiff(dimnames(x), ("j", "l")));

julia> reshape(unname(y, (rname, "j", "l")), :, 4) *
       reshape(unname(x, ("j", "l", rname)), 4, :) ≈ I
true
```
"""
function MA.gram_eigh_full_with_pinv(
        a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...
    )
    return gram_eigh_full_with_pinv_nameddims(
        a, dimnames_codomain, dimnames_domain; kwargs...
    )
end
function gram_eigh_full_with_pinv_nameddims(
        a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...
    )
    codomain = name.(dimnames_codomain)
    domain = name.(dimnames_domain)
    x_unnamed, y_unnamed = TA.gram_eigh_full_with_pinv(
        unnamed(a), dimnames(a), codomain, domain; kwargs...
    )
    name_xy = uniquename(dimnames(a, 1))
    dimnames_x = (domain..., name_xy)
    dimnames_y = (name_xy, domain...)
    return nameddims(x_unnamed, dimnames_x), nameddims(y_unnamed, dimnames_y)
end

"""
    TensorAlgebra.MatrixAlgebra.sqrth_safe(a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...) -> p

Square root of a named array `a`, interpreting it as an approximately
Hermitian positive semi-definite linear map from the domain to the codomain
dimension names. The result carries the same dimension names as `a`.
Eigenvalues below tolerance are clamped to zero.

`kwargs` are forwarded to `TensorAlgebra.sqrth_safe` on the underlying
unnamed array (e.g. `atol`, `rtol`).

See also [`TensorAlgebra.MatrixAlgebra.invsqrth_safe`](@ref) and
[`TensorAlgebra.MatrixAlgebra.sqrth_invsqrth_safe`](@ref).
"""
function MA.sqrth_safe(
        a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...
    )
    return sqrth_safe_nameddims(a, dimnames_codomain, dimnames_domain; kwargs...)
end
function sqrth_safe_nameddims(
        a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...
    )
    codomain = name.(dimnames_codomain)
    domain = name.(dimnames_domain)
    p_unnamed = TA.sqrth_safe(unnamed(a), dimnames(a), codomain, domain; kwargs...)
    return nameddims(p_unnamed, (codomain..., domain...))
end

"""
    TensorAlgebra.MatrixAlgebra.invsqrth_safe(a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...) -> p

Pseudo-inverse square root of a named array `a`, interpreting it as an
approximately Hermitian positive semi-definite linear map from the domain
to the codomain dimension names. The result carries the same dimension
names as `a`. Eigenvalues below tolerance are clamped to zero
(Moore-Penrose convention).

`kwargs` are forwarded to `TensorAlgebra.invsqrth_safe` on the underlying
unnamed array (e.g. `atol`, `rtol`).

See also [`TensorAlgebra.MatrixAlgebra.sqrth_safe`](@ref) and
[`TensorAlgebra.MatrixAlgebra.sqrth_invsqrth_safe`](@ref).
"""
function MA.invsqrth_safe(
        a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...
    )
    return invsqrth_safe_nameddims(a, dimnames_codomain, dimnames_domain; kwargs...)
end
function invsqrth_safe_nameddims(
        a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...
    )
    codomain = name.(dimnames_codomain)
    domain = name.(dimnames_domain)
    p_unnamed = TA.invsqrth_safe(unnamed(a), dimnames(a), codomain, domain; kwargs...)
    return nameddims(p_unnamed, (codomain..., domain...))
end

"""
    TensorAlgebra.MatrixAlgebra.sqrth_invsqrth_safe(a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...) -> p, pinv

Square root and pseudo-inverse square root of a named array `a` (see
`TensorAlgebra.MatrixAlgebra.sqrth_safe` and
`TensorAlgebra.MatrixAlgebra.invsqrth_safe`), from a single
eigendecomposition. Both results carry the same dimension names as `a`.

`kwargs` are forwarded to `TensorAlgebra.sqrth_invsqrth_safe` on the underlying
unnamed array (e.g. `atol`, `rtol`).
"""
function MA.sqrth_invsqrth_safe(
        a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...
    )
    return sqrth_invsqrth_safe_nameddims(a, dimnames_codomain, dimnames_domain; kwargs...)
end
function sqrth_invsqrth_safe_nameddims(
        a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...
    )
    codomain = name.(dimnames_codomain)
    domain = name.(dimnames_domain)
    p_unnamed, pinv_unnamed = TA.sqrth_invsqrth_safe(
        unnamed(a), dimnames(a), codomain, domain; kwargs...
    )
    dimnames_p = (codomain..., domain...)
    return nameddims(p_unnamed, dimnames_p), nameddims(pinv_unnamed, dimnames_p)
end

"""
    MatrixAlgebraKit.project_hermitian(a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...) -> h

Hermitian part `(m + m') / 2` of a named array `a`, interpreting it as a
linear map `m` from the domain to the codomain dimension names. The result
carries the same dimension names as `a`.
"""
function MAK.project_hermitian(
        a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...
    )
    return project_hermitian_nameddims(a, dimnames_codomain, dimnames_domain; kwargs...)
end
function project_hermitian_nameddims(
        a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...
    )
    codomain = name.(dimnames_codomain)
    domain = name.(dimnames_domain)
    h_unnamed = TA.project_hermitian(unnamed(a), dimnames(a), codomain, domain; kwargs...)
    return nameddims(h_unnamed, (codomain..., domain...))
end

"""
    Base.one(a::AbstractNamedTensor, dimnames_codomain, dimnames_domain) -> Id

Return an identity-operator-shaped named array sharing `a`'s dimension names,
codomain/domain partition, and element type. The fused codomain and domain sizes
must match. `a` is treated as a shape prototype and is not mutated.

The identity acts as the multiplicative identity for `ITensorBase.apply`: it
contracts on the domain names and renames the resulting codomain names back to
the domain names, leaving the input unchanged.

# Examples

```jldoctest
julia> using ITensorBase: apply, namedoneto, operator

julia> i, j, k, l = namedoneto.((2, 3, 2, 3), ("i", "j", "k", "l"));

julia> a = randn(i, j, k, l);

julia> Id = operator(one(a, (i, j), (k, l)), ("i", "j"), ("k", "l"));

julia> v = randn(k, l);

julia> apply(Id, v) ≈ v
true
```
"""
function Base.one(
        a::AbstractNamedTensor, dimnames_codomain, dimnames_domain
    )
    return one_nameddims(a, dimnames_codomain, dimnames_domain)
end
function one_nameddims(
        a::AbstractNamedTensor, dimnames_codomain, dimnames_domain
    )
    codomain = name.(dimnames_codomain)
    domain = name.(dimnames_domain)
    raw = TA.one(unnamed(a), dimnames(a), codomain, domain)
    return nameddims(raw, (codomain..., domain...))
end

const MATRIX_FUNCTIONS = [
    :exp, :cis, :log, :sqrt, :cbrt, :cos, :sin, :tan, :csc, :sec, :cot, :cosh, :sinh,
    :tanh,
    :csch, :sech, :coth, :acos, :asin, :atan, :acsc, :asec, :acot, :acosh, :asinh,
    :atanh,
    :acsch, :asech, :acoth,
]

for f in MATRIX_FUNCTIONS
    f_nameddims = Symbol(f, "_nameddims")
    @eval begin
        function Base.$f(
                a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...
            )
            return $f_nameddims(a, dimnames_codomain, dimnames_domain; kwargs...)
        end
        function $f_nameddims(
                a::AbstractNamedTensor, dimnames_codomain, dimnames_domain; kwargs...
            )
            codomain = name.(dimnames_codomain)
            domain = name.(dimnames_domain)
            fa_unnamed = TA.$f(
                unnamed(a), dimnames(a), codomain, domain; kwargs...
            )
            return nameddims(fa_unnamed, (codomain..., domain...))
        end
    end
end

#
# Projection into a symmetry-restricted named tensor.
#

"""
    TensorAlgebra.project(a::AbstractArray, codomain_inds, domain_inds; kwargs...) -> t
    TensorAlgebra.project(a::AbstractArray, inds; kwargs...) -> t

Build a named tensor from the dense array `a` by projecting it into the
symmetry-restricted space described by the indices, verifying that only a
negligible component of `a` is discarded and throwing an `InexactError`
otherwise (keyword arguments are forwarded to the `isapprox` tolerance
check). The three-argument form takes an explicit codomain/domain split (an
operator); the two-argument form takes a flat list of indices (a state, i.e.
an empty domain). The index axes select the backend: dense ranges give an
`Array`, graded ranges a block-sparse array, and TensorKit spaces a
`TensorMap`. `a` is indexed positionally in the order
`(codomain_inds..., domain_inds...)`.

`TensorAlgebra.tryproject` returns `nothing` instead of throwing, and
`TensorAlgebra.unchecked_project` skips the verification.
"""
TA.project

# Strip to `a`'s axes, lower to the TensorAlgebra verb, and reattach the names (passing
# through a `nothing` from `tryproject`). Two split entries per verb read the index type from
# whichever side is non-empty (an empty codomain is the all-domain case, the mirror of the
# empty-domain state), so neither an empty codomain nor an empty domain falls through to the
# unnamed-axis generic; the flat (state) form forwards to the split form.
for f in (:project, :tryproject, :unchecked_project)
    @eval begin
        function TA.$f(
                a::AbstractArray,
                codomain_inds::Tuple{NamedUnitRange, Vararg{NamedUnitRange}},
                domain_inds::Tuple{Vararg{NamedUnitRange}}; kwargs...
            )
            raw = TA.$f(a, unnamed.(codomain_inds), unnamed.(domain_inds); kwargs...)
            isnothing(raw) && return nothing
            return nameddims(raw, (name.(codomain_inds)..., name.(domain_inds)...))
        end
        function TA.$f(
                a::AbstractArray,
                codomain_inds::Tuple{},
                domain_inds::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}; kwargs...
            )
            raw = TA.$f(a, (), unnamed.(domain_inds); kwargs...)
            isnothing(raw) && return nothing
            return nameddims(raw, name.(domain_inds))
        end
        function TA.$f(
                a::AbstractArray, inds::Tuple{NamedUnitRange, Vararg{NamedUnitRange}};
                kwargs...
            )
            return TA.$f(a, inds, (); kwargs...)
        end
    end
end
