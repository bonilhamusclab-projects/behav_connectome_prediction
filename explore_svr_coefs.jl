using DataFrames
using Lazy
using MLBase
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


function get_Xy_mat(m::MeasureGroup, target::Target;
                region::Region=full_brain,
                subject_group::SubjectGroup=all_subjects)
  get_Xy_mat(DataInfo(m, target, subject_group, region))
end


function pred_diff(m::MeasureGroup)
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
                        m::MeasureGroup,
                        region::Region,
                        subject_group::SubjectGroup,
                        train_ratios::AbstractVector{Float64}=1./6:1./6:1.;
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


function learning_curve(m::MeasureGroup;
                        region::Region=left_select,
                        C=1.0,
                        subject_group::SubjectGroup=all_subjects,
                        seed::Nullable{Int}=Nullable(1234))
  isnull(seed) || srand(get(seed))

  di = DataInfo(m, diff_wpm, subject_group, region)

  svr = LinearSVR(C=C)
  lc_cvg_gen = cvg_gen_gen(RandomSub, 25)
  learning_curve(svr, lc_cvg_gen, m, region, subject_group)
end


function calc_coefs(d::DataInfo,
                    C::Float64;
                    n_folds::Int64=25,
                    n_perms::Int64=100)
  X::Matrix{Float64}, y::Vector{Float64} = get_Xy_mat(d)

  num_samples::Int64 = length(y)

  svr = LinearSVR(C=C)
  cvg::CrossValGenerator = cvg_gen(RandomSub, n_folds, num_samples)

  pos_perm_coefs = zeros(Float64, n_perms)
  neg_perm_coefs = zeros(Float64, n_perms)
  test_scores = zeros(Float64, n_perms)
  edges = get_edges(m, region)

  function fit_and_test_gen(ys::Vector{Float64},
                            fit_callback::Function)

    fit(inds::Vector{Int64}) = begin
      svr[:fit](X[inds, :], ys[inds])
      fit_callback()
      svr
    end

    test(c::PyObject, inds::Vector{Int64}) =
      r2_score(ys[inds], c[:predict](X[inds, :]))

    (fit, test)
  end

  for p::Int64 in 1:n_perms
    y_shuf = shuffle(y)
    pos_perm_coefs[p] = -Inf
    neg_perm_coefs[p] = Inf

    fit, test = begin
      fit_callback() = begin
        pos_perm_coefs[p] = max(pos_perm_coefs[p], maximum(svr[:coef_]))
        neg_perm_coefs[p] = min(neg_perm_coefs[p], minimum(svr[:coef_]))
      end
      fit_and_test_gen(y_shuf, fit_callback)
    end

    test_scores[p] = mean(cross_validate(fit, test, num_samples, cvg))

  end

  actual_coefs_mat::Matrix{Float64} = zeros(Float64, n_folds, length(edges))

  fit, test = begin
    state = Dict(:call_num=>0)
    fit_callback() = begin
      state[:call_num] = state[:call_num] + 1
      actual_coefs_mat[state[:call_num], :] = svr[:coef_]
    end
    fit_and_test_gen(y, fit_callback)
  end

  actual_score=mean(cross_validate(fit, test, num_samples, cvg))

  perm_info::DataFrame = begin
    perm_scores = sort(test_scores, rev=true)
    actual_score_rank=searchsortedfirst(perm_scores, actual_score, rev=true)

    DataFrame(perm_score=perm_scores,
              actual_score=actual_score,
              actual_score_rank=actual_score_rank,
              pos_perm_coef=sort(pos_perm_coefs, rev=true),
              neg_perm_coef=sort(neg_perm_coefs))
  end

  edge_info::DataFrame = begin
    ei = DataFrame()
    ei[:edge] = edges
    ei[:edge_name] = map(mk_edge_string, ei[:edge])
    ei[:coef] = mean(actual_coefs_mat, 1)[:]::Vector{Float64}

    ei[:pos_rank] = [searchsortedfirst(perm_info[:pos_perm_coef], c, rev=true)
                     for c in ei[:coef]]
    ei[:neg_rank] = [searchsortedfirst(perm_info[:neg_perm_coef], c)
                     for c in ei[:coef]]
    sort(ei, cols=:coef)
  end

  Dict(:edge_info => edge_info,
       :perm_info => perm_info)

end


function calc_all_coefs()
  cs = Dict(adw => Dict(left_select => 1e-2),
            atw => Dict(left_select => 1e-2))

  s_groups = SubjectGroup[all_subjects, improved, poor_pd]

  ret = Dict()
  for g in s_groups
    println(g)
    ret[g] = Dict()
    for m in keys(cs)
      ret[g][m] = Dict()
      for r in keys(cs[m])
        ret[g][m][r] = calc_coefs(m, cs[m][r], diff_wpm, r, group=g)
      end
    end
  end

  ret

end


function save_calc_coefs(calc_all_coefs_ret::Dict)
  for s::SubjectGroup in keys(ret)
    dir = joinpath("data/step4/svr/", "$s")
    isdir(dir) || mkpath(dir)
    for m::MeasureGroup in keys(ret[s])
      for r::Region in keys(ret[s][m])
        for k::Symbol in keys(ret[s][m][r])
          f_name = "$(m)_$(r)_$(k)s.csv"
          println(f_name)
          writetable(joinpath(dir, f_name), ret[s][m][r][k])
        end
      end
    end
  end
end
