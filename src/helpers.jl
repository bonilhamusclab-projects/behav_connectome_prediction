using DataFrames
using HypothesisTests
using Lazy
using Memoize

@enum Outcome adw atw

@enum Target diff_wpm se diff_wps

@enum Region full_brain left left_select left_select2

@enum SubjectGroup all_subjects improved poor_pd poor_pd_1

@enum DataSet conn lesion

@memoize function isEdge(edge_col::Symbol, region::Region)
  if !contains("$(edge_col)", "_to_")
    return false
  end

  names = @switch region begin
    left_select2; jhu_left_select_names2();
    left_select; jhu_left_select_names();
    left; jhu_left_names();
    full_brain; jhu_names();
  end

  a, b = edgeColToRoiNames(edge_col)
  in(a, names) & in(b, names)
end

@memoize isEdge(edge_col::Symbol) = isEdge(edge_col, full_brain)

@memoize isLeftEdge(edge_col::Symbol) = isEdge(edge_col, left)

@memoize isLeftSelectEdge(edge_col::Symbol) = isEdge(edge_col, left_select)

@memoize isLeftSelect2Edge(edge_col::Symbol) = isEdge(edge_col, left_select2)


@memoize isRoi(col::Symbol, names=jhu_names()) = in(col, names)

@memoize isLeftRoi(col::Symbol) = isRoi(col, jhu_left_names())

@memoize isLeftSelectRoi(col::Symbol) = isRoi(
  col, jhu_left_select_names())

@memoize isLeftSelect2Roi(col::Symbol) = isRoi(
  col, jhu_left_select_names2())


@memoize function regionFilter(r::Region, d::DataSet, predictors::Vector{Symbol})
  camelify(w::ASCIIString) = replace(w, r"_(.)", l -> uppercase(l[2]))
  filter_fn::Function = begin
    fn_string = "is$(r == full_brain ? "" : "_$r")_$(d == conn ? "edge" : "roi")"
    eval(parse(camelify(fn_string)))
  end

  filter(filter_fn, predictors)
end


macro eval_str(s)
  quote
    eval(parse($s))
  end
end


function getPredictors(o::Outcome, r::Region, d::DataSet)
  full::DataFrame = getFull(d, o)
  regionFilter(r, d, names(full))
end

data_dir() = joinpath(dirname(pwd()), "data")

@memoize getFull(d::DataSet, o::Outcome) = readtable("$(data_dir())/step3/$d/full_$o.csv")

namesSet(df_fn::Function) = @>> df_fn()[:name] Set map(symbol)

@memoize jhu() = readtable("$(data_dir())/jhu_coords.csv")
@memoize jhu_names() = jhu |> namesSet

@memoize jhu_left() = readtable("$(data_dir())/jhu_rois_left.csv")
@memoize jhu_left_names() = jhu_left |> namesSet

@memoize jhu_left_select() = readtable("$(data_dir())/jhu_rois_left_adjusted.csv")
@memoize jhu_left_select_names() = jhu_left_select |> namesSet

@memoize jhu_left_select2() = readtable("$(data_dir())/jhu_rois_left_adjusted2.csv")
@memoize jhu_left_select_names2() = jhu_left_select2 |> namesSet


@memoize edgeColToRoiNames(edge::Symbol) = @>> split("$edge", "_to_") map(symbol)


@memoize function mkEdgeString(edge::Symbol)
  left, right = edgeColToRoiNames(edge)
  "$(left) -- $(right)"
end


@memoize function mkPredictorStringGen(d::DataSet)
  @switch d begin
    conn; mkEdgeString;
    lesion; identity;
  end
end


function createPredictorNames(predictors::Vector{Symbol}, d::DataSet)
  mkPredString::Function = mkPredictorStringGen(d)
  map(mkPredString, predictors)
end


@memoize getTargetCol(t::Target, m::Outcome) = if t == se
  symbol("$(t)_$m")
else
  symbol("$(m)_$t")
end


@memoize function subjectFilterGenGen(s::SubjectGroup)
  @switch s begin
    all_subjects; m::Outcome -> d::DataFrame -> repmat([true], size(d, 1))
    improved; m::Outcome -> d::DataFrame -> d[symbol(m, "_diff_wpm")] .> 0
    poor_pd; m::Outcome -> d::DataFrame -> d[symbol("pd_", m, "_z")] .< 0
    poor_pd_1; m::Outcome -> d::DataFrame -> d[symbol("pd_", m, "_z")] .< 1
  end
end


@memoize subjectFilterGen(s::SubjectGroup, m::Outcome) =
  subjectFilterGenGen(s)(m)


@memoize subjectFilter(s::SubjectGroup, m::Outcome, df::DataFrame) = begin
  subjectFilterGenGen(s)(m)(df)
end

getCovarsForTarget(t::Target, m::Outcome) = if t == se
  Symbol[symbol("pd_$(m)")]
else
  Symbol[]
end


typealias HtInfo Dict{Symbol, Float64}

function htInfo(ht::HypothesisTests.HypothesisTest)
  ret = Dict{Symbol, Float64}()

  for tail in [:right, :left, :both]
    p_sym = symbol(tail, :_p)
    ret[p_sym] = pvalue(ht, tail=tail)
  end

  ret[:t] = ht.t
  ret
end

htInfo(arr::Vector{Float64}) = htInfo(OneSampleTTest(arr))

#stricter than the 2-sample t-tests
htInfo(arr1::Vector{Float64}, arr2::Vector{Float64}) = htInfo(arr1 - arr2)
