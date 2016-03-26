using Base.Test

include("helpers.jl")

@test 5 == @eval_str "2 + 3"


atw_left_select_edges = get_edges(atw, left_select)

edge_count(roi_count::Int64) = (roi_count^2 - roi_count)/2

@test length(atw_left_select_edges) == edge_count(46)
@test atw_left_select_edges == get_edges(adw, left_select)


atw_left_edges = get_edges(atw, left)
@test length(atw_left_edges) == edge_count(95)
@test atw_left_edges == get_edges(adw, left)

@test length(jhu_left_select_names()) == 46
@test length(jhu_left_names()) == 95

jhu_names = names_set(jhu)
@test length(jhu_names) == 189

@test isempty(setdiff(jhu_left_select_names(), jhu_left_names()))
@test isempty(setdiff(jhu_left_names(), jhu_names))

@test mk_edge_string(:x0_1) == "SFG_L -- SFG_R"


atw_num_subjects = size(full_atw(), 1)
adw_num_subjects = size(full_adw(), 1)

all_filter_gen = subject_filter_gen_gen(all_subjects)
all_atw_filter, all_adw_filter = map(all_filter_gen, [atw, adw])

@test sum(all_atw_filter(full_atw())) == atw_num_subjects
@test sum(all_adw_filter(full_adw())) == adw_num_subjects


improved_filter_gen = subject_filter_gen_gen(improved)
improved_atw_filter, improved_adw_filter = map(improved_filter_gen, [atw, adw])

@test improved_atw_filter(full_atw()) == (full_atw()[:atw_diff_wpm] .> 0)
@test improved_adw_filter(full_adw()) == (full_adw()[:adw_diff_wpm] .> 0)


poor_pd_filter_gen = subject_filter_gen_gen(poor_pd)
poor_pd_atw_filter, poor_pd_adw_filter = map(poor_pd_filter_gen, [atw, adw])

@test poor_pd_atw_filter(full_atw()) == (full_atw()[:pd_atw_z] .< 0)
@test poor_pd_adw_filter(full_adw()) == (full_adw()[:pd_adw_z] .< 0)


@test target_col(diff_wpm, atw) == :atw_diff_wpm
@test target_col(diff_wpm, adw) == :adw_diff_wpm
@test target_col(se, atw) == :se_atw
@test target_col(se, adw) == :se_adw
