macro data_include(f)
  quote
    f_name = $f
    src_dir = joinpath(dirname(pwd()), "src")
    include("$src_dir/$f_name")
  end
end

@data_include("helpers.jl")


edge_count(roi_count::Int64) = round(Int64, (roi_count^2 - roi_count)/2)

macro test_all_combos()
  quote
    covars = Dict{Target, Dict{Outcome, Vector{Symbol}}}(
      se => Dict( atw => [:pd_atw], adw => [:pd_adw]),
      diff_wpm => Dict( atw => [], adw => []))
    num_subjects = Dict(atw =>
                        Dict(all_subjects => size(get_full(conn, atw), 1),
                             improved => sum(get_full(conn, atw)[:atw_diff_wpm] .> 0),
                             poor_pd => sum(get_full(conn, atw)[:pd_atw_z] .< 0)),
                        adw =>
                        Dict(all_subjects => size(get_full(conn, adw), 1),
                             improved => sum(get_full(conn, adw)[:adw_diff_wpm] .> 0),
                             poor_pd => sum(get_full(conn, adw)[:pd_adw_z] .< 0))
                        )
    num_edges = Dict(full_brain => edge_count(189),
                     left => edge_count(95),
                     left_select => edge_count(46))

    for (measure::Outcome, subjects::Dict) in num_subjects
      for (subject::SubjectGroup, s_count::Int64) in subjects
        for (region::Region, e_count::Int64) in num_edges
          for (target::Target, cs::Dict{Outcome, Vector{Symbol}}) in covars
            test_fn(measure, subject, s_count, region, e_count, target, cs[measure])
          end
        end
      end
    end
  end

end
