using Base.Test

include("DataInfo.jl")
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

test_get_data(measure::MeasureGroup,
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

test_fn = test_get_data

@test_all_combos


test_get_Xy_mat(measure::MeasureGroup,
        subject_group::SubjectGroup,
        s_count::Int64,
        region::Region,
        e_count::Int64,
        target::Target,
        covars::Vector{Symbol}) = begin
  di = DataInfo(measure, target, subject_group, region)
  X, y = get_Xy_mat(di)

  @test size(X) == (s_count, e_count)
  @test length(y) == s_count
  @test isa(X, Matrix{Float64})
  @test isa(y, Vector{Float64})
end

test_fn = test_get_Xy_mat


@test_all_combos


create_was_called() = Dict([ i => false for i in
                   [adw, atw,
                    diff_wpm, se,
                    all_subjects, improved, poor_pd, poor_pd_1,
                    full_brain, left_select, left] ])

was_called = create_was_called()

update_was_called(di::DataInfo) = begin
  for f in fieldnames(di)
    was_called[di.(f)] = true
  end
  di
end

for_all_combos(fn=update_was_called)

for k in keys(was_called)
  @test was_called[k]
end

was_called = create_was_called()

filter_vals = Dict(SubjectGroup => [all_subjects],
                   MeasureGroup => [adw],
                   Target => [diff_wpm],
                   Region => [left_select])

for_all_combos(fn=update_was_called,
               subject_groups=filter_vals[SubjectGroup],
               measure_groups=filter_vals[MeasureGroup],
               targets=filter_vals[Target],
               regions=filter_vals[Region]
               )

for k in keys(was_called)
  expected = in(k, filter_vals[typeof(k)])
  @test was_called[k] == expected
end
