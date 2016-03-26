using Base.Test

include("explore_linear_reg_coefs.jl")
include("helpers.jl")



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


edge_count(roi_count::Int64) = round(Int64, (roi_count^2 - roi_count)/2)

covars = Dict{Target, Dict{MeasureGroup, Vector{Symbol}}}(
  se => Dict( atw => [:pd_atw], adw => [:pd_adw]),
  diff_wpm => Dict( atw => [], adw => []))
num_subjects = Dict(atw =>
                    Dict(all_subjects => size(full_atw(), 1),
                         improved => sum(full_atw()[:atw_diff_wpm] .> 0),
                         poor_pd => sum(full_atw()[:pd_atw_z] .< 0)),
                    adw =>
                    Dict(all_subjects => size(full_adw(), 1),
                         improved => sum(full_adw()[:adw_diff_wpm] .> 0),
                         poor_pd => sum(full_adw()[:pd_adw_z] .< 0))
                    )
num_edges = Dict(full_brain => edge_count(189),
                 left => edge_count(95),
                 left_select => edge_count(46))

for (measure::MeasureGroup, subjects::Dict) in num_subjects
  for (subject::SubjectGroup, s_count::Int64) in subjects
    for (region::Region, e_count::Int64) in num_edges
      for (target::Target, cs::Dict{MeasureGroup, Vector{Symbol}}) in covars
        di = DataInfo(measure, target, subject, region)
        println(to_string(di))
        target_col = symbol(target, "_", measure)
        test_get_data(di, s_count, target_col, cs[measure], e_count)
      end
    end
  end
end
