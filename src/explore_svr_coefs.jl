using Colors
using DataFrames
using HypothesisTests
using Lazy
using MLBase
using PValueAdjust
using PyCall

include("DataInfo.jl")
include("helpers.jl")

@pyimport sklearn.svm as svm
LinearSVR = svm.LinearSVR


macro l2_from_true(x)
  :(sum( (y_true - $x).^2))
end

function r2_score(y_true::Vector{Float64}, y_pred::Vector{Float64})

  numerator::Float64 = @l2_from_true y_pred
  denominator::Float64 = @l2_from_true mean(y_true)

  1 - numerator/denominator
end


function get_Xy_mat(m::Outcome, target::Target;
                region::Region=full_brain,
                subject_group::SubjectGroup=all_subjects)
  get_Xy_mat(DataInfo(m, target, subject_group, region))
end


function pred_diff(m::Outcome)
  svr = LinearSVR()

  X, y = get_Xy_mat(m, diff_wpm)

  fit(inds::Vector{Int64}) = svr[:fit](X[inds, :], y[inds])

  test(c::PyObject, inds::Vector{Int64}) =
    r2_score(y[inds], c[:predict](X[inds, :]))

  num_samples = length(y)

  cross_validate(fit,
                 test,
                 num_samples,
                 Kfold(num_samples, 5))
end


function num_samples(cv::CrossValGenerator)
  @switch isa(cv, _) begin
    Kfold; length(cv.permseq)
    StratifiedRandomSub; sum(map(length, cv.idxs))
    cv.n
  end
end


function ratios_to_counts(ratios::AbstractVector{Float64}, n::Int64)
  @assert all( (ratios .<= 1.) &  (ratios .>= 0.) )

    get_count(r::Float64) = round(Int, r*n)::Int64

    map(get_count, ratios)
end


function learning_curve(svr::PyObject,
                        lc_cvg_gen::Function,
                        m::Outcome,
                        region::Region,
                        subject_group::SubjectGroup,
                        train_ratios::AbstractVector{Float64};
                        score_fn::Function=r2_score)

  X, y = get_Xy_mat(m, diff_wpm, region=region, subject_group=subject_group)
  num_samples::Int64 = length(y)

  train_sizes::Vector{Int64} = ratios_to_counts(train_ratios, num_samples)
  train_sizes = train_sizes[train_sizes .> 2]

  ret::DataFrame = begin
    mk_zeros = () -> zeros(Float64, length(train_sizes))

    DataFrame(train_mn=mk_zeros(),
              train_std=mk_zeros(),
              test_mn=mk_zeros(),
              test_std=mk_zeros(),
              train_size=train_sizes)
  end

  test(c::PyObject, inds::Vector{Int64}) =
    score_fn(y[inds], c[:predict](X[inds, :]))

  for (ix, t) in enumerate(train_sizes)
    cvg::CrossValGenerator = lc_cvg_gen(t)

    train_scores = Float64[]

    fit(inds::Vector{Int64}) = begin
      svr[:fit](X[inds, :], y[inds])
      train_score = test(svr, inds)
      train_scores = [train_scores; train_score]
      svr
    end

    test_scores = cross_validate(fit, test, t, cvg)

    ret[ix, :train_mn] = mean(train_scores)
    ret[ix, :train_std] = std(train_scores)
    ret[ix, :test_mn] = mean(test_scores)
    ret[ix, :test_std] = std(test_scores)
  end

  ret
end


function cvg_gen_gen(cvgT::Type, n_folds::Int64)
  @assert cvgT <: CrossValGenerator
  @switch cvgT begin
    Kfold; (n_samples::Int64) -> Kfold(n_samples, n_folds)
    RandomSub; (n_samples::Int64) -> RandomSub(n_samples, round(Int64, .8 * n_samples), n_folds)
  end
end

cvg_gen(cvgT::Type, n_folds::Int64, n_samples::Int64) = cvg_gen_gen(cvgT, n_folds)(n_samples)


function learning_curve(m::Outcome;
                        region::Region=left_select,
                        C=1.0,
                        subject_group::SubjectGroup=all_subjects,
                        seed::Nullable{Int}=Nullable(1234),
                        train_ratios::AbstractVector{Float64}=1./6:1./6:1.)
  isnull(seed) || srand(get(seed))

  svr = LinearSVR(C=C)
  lc_cvg_gen = cvg_gen_gen(RandomSub, 25)
  learning_curve(svr, lc_cvg_gen, m, region, subject_group, train_ratios)
end


function learning_curve(di::DataInfo, C::Float64; seed::Nullable{Int}=Nullable(1234),
                        train_ratios::AbstractVector{Float64}=1./6:1./6:1.)
  learning_curve(di.outcome, region=di.region, C=C, subject_group=di.subject_group, seed=seed,
                 train_ratios=train_ratios)
end


function calc_coefs(d::DataInfo,
                    C::Float64;
                    n_perms::Int64=1000,
                    seed::Nullable{Int}=Nullable(1234))

  isnull(seed) || srand(get(seed))

  X::Matrix{Float64}, y::Vector{Float64} = get_Xy_mat(d)

  num_samples::Int64 = length(y)

  svr = LinearSVR(C=C)
  cvg::CrossValGenerator = cvg_gen(RandomSub, n_perms, num_samples)

  edges = get_edges(d)
  n_edges = length(edges)

  coefs::DataFrame = begin
    ret = DataFrame()
    for e in edges
      ###Hack to keep it float64 while being NA
      ret[e] = repmat([-Inf], n_perms)
      ret[ret[e] .== -Inf, :] = NA
    end
    ret
  end

  fit, test = begin

    state = Dict(:fit_call => 0)

    fit_fn(inds::Vector{Int64}) = begin
      state[:fit_call] += 1
      println(state[:fit_call])
      svr[:fit](X[inds, :], y[inds])
      for (ix, c) in enumerate(svr[:coef_])
        coefs[state[:fit_call], edges[ix]] = c
      end
      svr
    end

    test_fn(c::PyObject, inds::Vector{Int64}) =
      r2_score(y[inds], c[:predict](X[inds, :]))

    (fit_fn, test_fn)
  end

  test_scores::Vector{Float64} = cross_validate(fit, test, num_samples, cvg)

  typealias HtInfo Dict{Symbol, Float64}
  ht_info(arr::Vector{Float64}) = begin
    ret = Dict{Symbol, Float64}()
    ht = OneSampleTTest(arr)

    for tail in [:right, :left, :both]
      p_sym = symbol(tail, :_p)
      ret[p_sym] = pvalue(ht, tail=tail)
    end

    ret[:t] = ht.t
    ret
  end

  pv(arr::Vector{Float64}, tail::Symbol=:both) = pvalue(OneSampleTTest(arr), tail=tail)

  pred_info::DataFrame = begin
    ht = ht_info(test_scores)
    pi = DataFrame(
      mean=mean(test_scores),
      std=std(test_scores),
      t=ht[:t],
      right_p=ht[:right_p],
      left_p=ht[:left_p],
      both_p=ht[:both_p],
      num_perms=n_perms)
  end

  edge_info::DataFrame = begin
    ei = DataFrame()
    ei[:edge] = edges
    ei[:edge_name] = map(mk_edge_string, ei[:edge])


    edge_apply(fn::Function) = [fn(dropna(coefs[e])) for e in edges]

    ei[:mean] = edge_apply(mean)
    ei[:std] = edge_apply(std)

    hts::Vector{HtInfo} = edge_apply(ht_info)
    for k::Symbol in keys(hts[1])
      ei[k] = Float64[i[k] for i in hts]

      is_p_measure::Bool = endswith(string(k), "_p")
      if is_p_measure
        ei[symbol(k, :_adj)]= padjust(ei[k], BenjaminiHochberg)
      end
    end

    sort(ei, cols=:t)
  end

  Dict(:edge_info => edge_info,
       :pred_info => pred_info)

end


function calc_all_coefs(cs::Dict{DataInfo, Float64})

  ret = Dict{DataInfo, Dict{Symbol, DataFrame}}()
  for (di::DataInfo, C::Float64) in cs
    println(di)
    ret[di] = calc_coefs(di, C)
  end

  ret

end


function save_calc_coefs(calc_all_coefs_ret::Dict{DataInfo, Dict{Symbol, DataFrame}})
  for di::DataInfo in keys(calc_all_coefs_ret)
    dir = joinpath("data/step4/svr/")
    isdir(dir) || mkpath(dir)
    for (k::Symbol, data::DataFrame) in ret[di]
      f_name = "$(di)_$(k)s.csv"
      println(f_name)
      writetable(joinpath(dir, f_name), data)
    end
  end
end


function lc_flip(di::DataInfo, C::Float64, flip_ratio::Float64)
  ret = DataFrame()
  for C_tmp = [C/flip_ratio, C, C*flip_ratio]

    suff(k::Symbol) = if(C_tmp != C)
      C_tmp > C ? symbol(k, "_gt") : symbol(k, "_lt") else
      k end

    lc = learning_curve(di, C_tmp, train_ratios=5./6:1./6:1.)[end, :]
    ret[suff(:C)] = C_tmp

    for n in names(lc)
      ret[suff(n)] = lc[n]
    end

  end

  for f in fieldnames(di)
    ret[f] = di[f]
  end

  ret
end


###Hypothesis: only brain region determines optimal C for a measure
function param_test()
  best_Cs = Dict(
    atw => Dict(
      left => 5e-4,
      left_select => 5e-3,
      full_brain => 5e-4
      ),
    adw => Dict(
      left => 5e-4,
      left_select => 5e-3,
      full_brain => 5e-4
      )
    )

  specials = Dict(
    DataInfo(adw, diff_wpm, improved, full_brain) => 1e-3
    )

  data_infos = for_all_combos(targets=[diff_wpm])

  ret = DataFrame()

  for di in data_infos
    println()
    println("#####")

    C = in(di, keys(specials)) ? specials[di] : best_Cs[di.outcome][di.region]

    println("di: $di, C: $C")
    lc::DataFrame = lc_flip(di, C, 5.)
    ret = isempty(ret) ? lc : vcat(ret, lc)

    println()
  end

  sort(ret, cols=fieldnames(DataInfo))
end


function plot_param_test(pt_res::DataFrame)
  function mk_layer_di(col::Symbol, clr::AbstractString)
    col_std::Symbol = replace("$col", "_mn", "_std")
    ymax = pt_res[col] + pt_res[col_std]
    ymin = pt_res[col] - pt_res[col_std]
    di_strings = [to_string(DataInfo(m[2], diff_wpm, s[2], r[2]))[1:end-9]
                  for (m, s, r) in eachrow(pt_res[[:outcome, :subject_group, :region]])]

    layer(x, x=di_strings, y=col, ymax=ymax, ymin=ymin,
          Geom.line, Geom.point,
          Theme(default_color=parse(Colorant, clr))
          )
  end

  plot_test() = plot(mk_layer_di(:test_mn, "#00FF00"),
                     mk_layer_di(:test_mn_lt, "#FF0000"),
                     mk_layer_di(:test_mn_gt, "#0000FF"))

  plot_train() = plot(mk_layer_di(:train_mn, "#00FF00"),
                      mk_layer_di(:train_mn_lt, "#FF0000"),
                      mk_layer_di(:train_mn_gt, "#0000FF"))

  (plot_test, plot_train)

end
