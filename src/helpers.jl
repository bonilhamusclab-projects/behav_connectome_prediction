using DataFrames
using Lazy
using Memoize

@enum Outcome adw atw

@enum Target diff_wpm se

@enum Region full_brain left left_select

@enum SubjectGroup all_subjects improved poor_pd poor_pd_1

@enum DataSet conn lesion

@memoize function is_left_edge(edge_col::Symbol)
  a, b = edge_col_to_roi_names(edge_col)
  in(a, jhu_left_names()) & in(b, jhu_left_names())
end


@memoize function is_left_select_edge(edge_col::Symbol)
  a, b = edge_col_to_roi_names(edge_col)
  in(a, jhu_left_select_names()) & in(b, jhu_left_select_names())
end


@memoize function roi_col_to_jhu_name(roi_col::Symbol)
  roi_reg = r"x([0-9]+)"
  m = match(roi_reg, string(roi_col))
  ix = parse(Int64, m[1]) + 1
  jhu()[ix, :name]
end


@memoize is_left_roi(roi_col::Symbol) = in(
  roi_col_to_jhu_name(roi_col), jhu_left_names())


@memoize is_left_select_roi(roi_col::Symbol) = in(
  roi_col_to_jhu_name(roi_col), jhu_left_select_names())


@memoize function region_filter(r::Region, d::DataSet, predictors::Vector{Symbol})

  filter_fn::Function = if(r == full_brain)
    p::Symbol -> true
  else
    fn_string = "is_$(r)_$(d == conn ? "edge" : "roi")"
    eval(parse(fn_string))
  end

  filter(filter_fn, predictors)
end


macro eval_str(s)
  quote
    eval(parse($s))
  end
end


function get_predictors(o::Outcome, r::Region, d::DataSet)
  full::DataFrame = get_full(d, o)

  predictor_cols::Vector{Symbol} = filter(
    n -> startswith(string(n), "x"),
    names(full))

  region_filter(r, d, predictor_cols)
end

data_dir() = joinpath(dirname(pwd()), "data")

@memoize get_full(d::DataSet, o::Outcome) = readtable("$(data_dir())/step3/$d/full_$o.csv")

@memoize jhu() = readtable("$(data_dir())/jhu_coords.csv")

names_set(df_fn::Function) = @> df_fn()[:name] Set

@memoize jhu_left() = readtable("$(data_dir())/jhu_rois_left.csv")
@memoize jhu_left_names() = @> jhu_left names_set

@memoize jhu_left_select() = readtable("$(data_dir())/jhu_rois_left_adjusted.csv")
@memoize jhu_left_select_names() = @> jhu_left_select names_set


edge_reg = r"x([0-9]+)_([0-9]+)"

@memoize function edge_col_to_x(edge::Symbol, x::Symbol)
  m = match(edge_reg, string(edge))
  map([1, 2]) do i
    ix = parse(Int64, m[i]) + 1
    jhu()[ix, x]
  end
end


@memoize function edge_col_to_roi_names(edge::Symbol)
  edge_col_to_x(edge, :name)
end


@memoize function mk_edge_string(edge::Symbol)
  left, right = edge_col_to_roi_names(edge)
  "$(left) -- $(right)"
end


@memoize function mk_predictor_string_gen(d::DataSet)
  @switch d begin
    conn; mk_edge_string;
    lesion; roi_col_to_jhu_name;
  end
end


function create_predictor_names(predictors::Vector{Symbol}, d::DataSet)
  mk_pred_string::Function = mk_predictor_string_gen(d)
  map(mk_pred_string, predictors)
end


@memoize function get_target_col(t::Target, m::Outcome)
  @switch t begin
    se; symbol("$(t)_$m")
    diff_wpm; symbol("$(m)_$t")
  end
end


@memoize function subject_filter_gen_gen(s::SubjectGroup)
  @switch s begin
    all_subjects; m::Outcome -> d::DataFrame -> repmat([true], size(d, 1))
    improved; m::Outcome -> d::DataFrame -> d[symbol(m, "_diff_wpm")] .> 0
    poor_pd; m::Outcome -> d::DataFrame -> d[symbol("pd_", m, "_z")] .< 0
    poor_pd_1; m::Outcome -> d::DataFrame -> d[symbol("pd_", m, "_z")] .< 1
  end
end


@memoize subject_filter_gen(s::SubjectGroup, m::Outcome) =
  subject_filter_gen_gen(s)(m)


@memoize subject_filter(s::SubjectGroup, m::Outcome, df::DataFrame) = begin
  subject_filter_gen_gen(s)(m)(df)
end


covars_for_target(t::Target, m::Outcome) = begin
  @switch t begin
    se; Symbol[symbol("pd_$(m)")]
    diff_wpm; Symbol[]
  end
end
