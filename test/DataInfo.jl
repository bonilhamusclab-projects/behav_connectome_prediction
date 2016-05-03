using Base.Test

include("test_utils.jl")

@data_include("DataInfo.jl")
@data_include("helpers.jl")


function testGetData(di::DataInfo,
                       expected_num_rows::Int64,
                       expected_target_col::Symbol,
                       expected_covars::Vector{Symbol},
                       expected_num_edges::Int64)
  data::DataFrame, edges::Vector{Symbol}, covars::Vector{Symbol}, target_col::Symbol =
    getData(di)

  @test length(edges) == expected_num_edges
  @test covars == expected_covars
  @test size(data) == (expected_num_rows, expected_num_edges + length(expected_covars) + 1)
end

testGetData(measure::Outcome,
        subject_group::SubjectGroup,
        s_count::Int64,
        region::Region,
        e_count::Int64,
        target::Target,
        covars::Vector{Symbol}) = begin
  di = DataInfo(measure, target, subject_group, region, conn)
  println(to_string(di))

  target_col = symbol(target, "_", measure)
  testGetData(di, s_count, target_col, covars, e_count)
end

test_fn = testGetData

@test_all_combos


testGetXyMat(measure::Outcome,
        subject_group::SubjectGroup,
        s_count::Int64,
        region::Region,
        e_count::Int64,
        target::Target,
        covars::Vector{Symbol}) = begin
  di = DataInfo(measure, target, subject_group, region, conn)
  X, y = getXyMat(di)

  @test size(X) == (s_count, e_count)
  @test length(y) == s_count
  @test isa(X, Matrix{Float64})
  @test isa(y, Vector{Float64})
end

test_fn = testGetXyMat


@test_all_combos


createWasCalled() = Dict([ i => false for i in
                   [adw, atw,
                    diff_wpm, se,
                    all_subjects, improved, poor_pd, poor_pd_1,
                    full_brain, left_select, left,
                    conn, lesion] ])

was_called = createWasCalled()

updateWasCalled(di::DataInfo) = begin
  for f in setdiff(fieldnames(di), [:include_covars])
    was_called[di.(f)] = true
  end
  di
end

forAllCombos(fn=updateWasCalled)

for k in keys(was_called)
  @test was_called[k]
end

was_called = createWasCalled()

filter_vals = Dict(SubjectGroup => [all_subjects],
                   Outcome => [adw],
                   Target => [diff_wpm],
                   Region => [left_select],
                   DataSet => [conn])

forAllCombos(fn=updateWasCalled,
               subject_groups=filter_vals[SubjectGroup],
               outcomes=filter_vals[Outcome],
               targets=filter_vals[Target],
               regions=filter_vals[Region],
               datasets=filter_vals[DataSet]
               )

for k in keys(was_called)
  expected = in(k, filter_vals[typeof(k)])
  @test was_called[k] == expected
end
