using Base.Test

include("explore_svr_coefs.jl")
include("helpers.jl")
include("test_utils.jl")


test_fn(measure::MeasureGroup,
        subject_group::SubjectGroup,
        s_count::Int64,
        region::Region,
        e_count::Int64,
        target::Target,
        covars::Vector{Symbol}) = begin
  X, y = get_Xy(measure, target;
                region=region,
                subject_group=subject_group)

  @test size(X) == (s_count, e_count)
  @test length(y) == s_count
  @test isa(X, Matrix{Float64})
  @test isa(y, Vector{Float64})
end


@test_all_combos