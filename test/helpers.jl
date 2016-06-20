using Base.Test
include("testUtils.jl")

@data_include("helpers.jl")

@test 5 == @eval_str "2 + 3"

getEdges(o::Outcome, r::Region) = getPredictors(o, r, conn)

atw_left_select_edges = getEdges(atw, left_select)

edgeCount(roi_count::Int64) = (roi_count^2 - roi_count)/2

@test length(atw_left_select_edges) == edgeCount(46)
@test atw_left_select_edges == getEdges(adw, left_select)


atw_left_edges = getEdges(atw, left)
@test length(atw_left_edges) == edgeCount(95)
@test atw_left_edges == getEdges(adw, left)

@test length(jhu_left_select_names()) == 46
@test length(jhu_left_names()) == 95

@test length(jhu_names()) == 189

@test isempty(setdiff(jhu_left_select_names(), jhu_left_names()))
@test isempty(setdiff(jhu_left_names(), jhu_names()))

@test mkEdgeString(:a_to_b) == "a -- b"


atw_num_subjects = size(getFull(conn, atw), 1)
adw_num_subjects = size(getFull(conn, adw), 1)

all_filter_gen = subjectFilterGenGen(all_subjects)
all_atw_filter, all_adw_filter = map(all_filter_gen, [atw, adw])

@test sum(all_atw_filter(getFull(conn, atw))) == atw_num_subjects
@test sum(all_adw_filter(getFull(conn, adw))) == adw_num_subjects


improved_filter_gen = subjectFilterGenGen(improved)
improved_atw_filter, improved_adw_filter = map(improved_filter_gen, [atw, adw])

@test improved_atw_filter(getFull(conn, atw)) == (getFull(conn, atw)[:atw_diff_wpm] .> 0)
@test improved_adw_filter(getFull(conn, adw)) == (getFull(conn, adw)[:adw_diff_wpm] .> 0)


poor_pd_filter_gen = subjectFilterGenGen(poor_pd)
poor_pd_atw_filter, poor_pd_adw_filter = map(poor_pd_filter_gen, [atw, adw])

@test poor_pd_atw_filter(getFull(conn, atw)) == (getFull(conn, atw)[:pd_atw_z] .< 0)
@test poor_pd_adw_filter(getFull(conn, adw)) == (getFull(conn, adw)[:pd_adw_z] .< 0)


@test getTargetCol(diff_wpm, atw) == :atw_diff_wpm
@test getTargetCol(diff_wpm, adw) == :adw_diff_wpm
@test getTargetCol(se, atw) == :se_atw
@test getTargetCol(se, adw) == :se_adw

@test getCovarsForTarget(se, atw) == [:pd_atw]
@test getCovarsForTarget(se, adw) == [:pd_adw]
@test getCovarsForTarget(diff_wpm, atw) == Symbol[]
@test getCovarsForTarget(diff_wpm, adw) == Symbol[]
