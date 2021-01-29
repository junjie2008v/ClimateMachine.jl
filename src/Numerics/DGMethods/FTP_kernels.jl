# This kernel computes the fast tensor-product x vector
# vout = (phit ⊗ phis ⊗ phir) * vin
#
@kernel function ftpxv_hex_kernel!(
    vin::AbstractArray{FT, 2},
    vin_den::Union{AbstractArray{FT, 2}, Nothing},
    vout::AbstractArray{FT, 2},
    phir::Union{AbstractArray{FT, 2}, Nothing},
    phis::Union{AbstractArray{FT, 2}, Nothing},
    phit::Union{AbstractArray{FT, 2}, Nothing},
    si::Int,
    sj::Int,
    sk::Int,
    sr::Int,
    ss::Int,
    st::Int,
    temp::AbstractArray{FT, 3},
    ::Val{d1m},
    ::Val{d2m},
) where {d1m, d2m, FT <: AbstractFloat}

    e = @index(Group, Linear)
    i1, i2 = @index(Local, NTuple)
    s_1 = @localmem FT (d1m, d2m)
    l_phir = @private FT (1)
    l_phis = @private FT (1)
    l_phit = @private FT (1)

    if !(phir === nothing) && i1 ≤ sr && i2 ≤ si
        l_phir[1] = phir[i1, i2]
    end

    if !(phis === nothing) && i1 ≤ ss && i2 ≤ sj
        l_phis[1] = phis[i1, i2]
    end

    if !(phit === nothing) && i1 ≤ st && i2 ≤ sk
        l_phit[1] = phit[i1, i2]
    end
    # Apply phir -----------------------------------------------------------
    if !(phir === nothing)
        @inbounds for k in 1:sk, j in 1:sj
            if i1 ≤ sr && i2 ≤ si
                ijk = i2 + ((j - 1) + (k - 1) * sj) * si
                if vin_den === nothing
                    @inbounds s_1[i1, i2] = l_phir[1] * vin[ijk, e]
                else
                    @inbounds s_1[i1, i2] =
                        l_phir[1] * vin[ijk, e] / vin_den[ijk, e]
                end
            end
            @synchronize
            if i1 ≤ sr && i2 == 1
                @inbounds for i in 2:si
                    s_1[i1, 1] += s_1[i1, i]
                end
            end
            @synchronize
            if i1 ≤ sr && i2 == 1
                ijk = i1 + ((j - 1) + (k - 1) * sj) * sr
                @inbounds temp[ijk, 1, e] = s_1[i1, 1]
            end
            @synchronize
        end
    else # in this case, phir is assumed to be an identity matrix
        @inbounds for k in 1:sk
            if i1 ≤ sr && i2 ≤ sj
                ijk = i1 + ((i2 - 1) + (k - 1) * sj) * sr
                if vin_den === nothing
                    @inbounds temp[ijk, 1, e] = vin[ijk, e]
                else
                    @inbounds temp[ijk, 1, e] = vin[ijk, e] / vin_den[ijk, e]
                end
            end
        end
        @synchronize
    end
    #-----------------------------------------------------------------------

    # Apply phis -----------------------------------------------------------
    if !(phis === nothing)
        @inbounds for k in 1:sk, r in 1:sr
            if i1 ≤ ss && i2 ≤ sj
                ijk = r + ((i2 - 1) + (k - 1) * sj) * sr
                @inbounds s_1[i1, i2] = l_phis[1] * temp[ijk, 1, e]
            end
            @synchronize
            if i1 ≤ ss && i2 == 1
                @inbounds for j in 2:sj
                    s_1[i1, 1] += s_1[i1, j]
                end
            end
            @synchronize
            if i1 ≤ ss && i2 == 1
                ijk = r + ((i1 - 1) + (k - 1) * ss) * sr
                @inbounds temp[ijk, 2, e] = s_1[i1, 1]
            end
            @synchronize
        end
    else # in this case, phis is assumed to be an identity matrix
        @inbounds for k in 1:sk
            if i1 ≤ sr && i2 ≤ ss
                ijk = i1 + ((i2 - 1) + (k - 1) * ss) * sr
                @inbounds temp[ijk, 2, e] = temp[ijk, 1, e]
            end
        end
        @synchronize
    end
    #-----------------------------------------------------------------------

    # Apply phit -----------------------------------------------------------
    if !(phit === nothing)
        @inbounds for s in 1:ss, r in 1:sr
            if i1 ≤ st && i2 ≤ sk
                ijk = r + ((s - 1) + (i2 - 1) * ss) * sr
                @inbounds s_1[i1, i2] = l_phit[1] * temp[ijk, 1, e]
            end
            @synchronize
            if i1 ≤ st && i2 == 1
                @inbounds for k in 2:sk
                    s_1[i1, 1] += s_1[i1, k]
                end
            end
            @synchronize
            if i1 ≤ st && i2 == 1
                ijk = r + ((s - 1) + (i1 - 1) * ss) * sr
                @inbounds vout[ijk, e] = s_1[i1, 1]
            end
            @synchronize
        end
    else # in this case, phit is assumed to be an identity matrix
        @inbounds for t in 1:st
            if i1 ≤ sr && i2 ≤ ss
                ijk = i1 + ((i2 - 1) + (t - 1) * ss) * sr
                @inbounds vout[ijk, e] = temp[ijk, 2, e]
            end
        end
        @synchronize
    end
    #-----------------------------------------------------------------------

end
