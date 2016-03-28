using Base.Test

include("explore_linear_reg_coefs.jl")
include("helpers.jl")
include("test_utils.jl")



function test_get_data(di::DataInfo,
                       expected_num_rows::Int64,
                       expected_target_col::Symbol,
                       expected_covars::Vector{Symbol},
                       expected_num_edges::Int64)
  data::DataFrame, edges::Vector{Symbol}, covars::Vector{Symbol}, target_col::Symbol =
    get_data(di)

  @test length(edges) == expected_num_edges
  @test covars == expected_covars
  @test size(data) == (expected_num_rows, expected_num_edges + length(expected_covars) + 1)
end

test_fn(measure::MeasureGroup,
        subject_group::SubjectGroup,
        s_count::Int64,
        region::Region,
        e_count::Int64,
        target::Target,
        covars::Vector{Symbol}) = begin
  di = DataInfo(measure, target, subject_group, region)
  println(to_string(di))

  target_col = symbol(target, "_", measure)
  test_get_data(di, s_count, target_col, covars, e_count)
end

@test_all_combos
