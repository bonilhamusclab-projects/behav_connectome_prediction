using DataFrames
using Lazy
using MLBase
using PyCall

include("helpers.jl")

@pyimport sklearn.svm as svm
LinearSVR = svm.LinearSVR


function r2_score(y_true::Vector{Float64}, y_pred::Vector{Float64})
  numerator::Float64 = sum( (y_true - y_pred).^2 )
  denominator::Float64 = begin
    y_mn = mean(y_true)
    sum( (y_true .- y_mn).^2 )
  end
  1 - numerator/denominator
end

@enum Target diff_wpm se


function get_Xy(t::Target, m::MeasureGroup;
                edge_filter::Function = edge::Symbol -> true)
  full::DataFrame = @eval_str "full_$m()"
  target::Symbol = begin
    str = @switch t begin
      diff_wpm; ":$(m)_diff_wpm"
      se; ":se_$m"
    end
    @eval_str str
  end

  X::Matrix{Float64} = begin
    edges::Vector{Symbol} = filter(edge_filter, get_edges(full))
    Matrix(full[:, edges])
  end
  y::Vector{Float64} = Array(full[target])
  X, y
end


function pred_diff(m::MeasureGroup)
  svr = LinearSVR()

  X, y = get_Xy(diff_wpm, m)

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
  @switch isa(x, _) begin
    Kfold; length(x.permseq)
    x.n
  end
end


function ratios_to_counts(ratios::AbstractVector{Float64}, n::Int64)
  @assert all( (ratios .<= 1.) &  (ratios .>= 0.) )

    get_count(r::Float64) = round(Int, r*n)::Int64

    map(get_count, ratios)
end


function learning_curve(svr::PyObject,
                        cv_gen::Function,
                        m::MeasureGroup,
                        edge_filter::Function,
                        train_ratios::AbstractVector{Float64}=1./6:1./6:1.;
                        score_fn::Function=r2_score)

  X, y = get_Xy(diff_wpm, m, edge_filter=edge_filter)
  num_samples::Int64 = length(y)

  train_sizes::Vector{Int64} = ratios_to_counts(train_ratios, num_samples)

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
    cv::CrossValGenerator = cv_gen(t)

    train_scores = Float64[]

    fit(inds::Vector{Int64}) = begin
      svr[:fit](X[inds, :], y[inds])
      train_score = test(svr, inds)
      train_scores = [train_scores; train_score]
      svr
    end

    test_scores = cross_validate(fit, test, t, cv)

    ret[ix, :train_mn] = mean(train_scores)
    ret[ix, :train_std] = std(train_scores)
    ret[ix, :test_mn] = mean(test_scores)
    ret[ix, :test_std] = std(test_scores)
  end

  ret
end


function learning_curve(m::MeasureGroup;
                        edge_filter::Function = is_left_hemi_select_edge,
                        C=1.0,
                        seed::Nullable{Int}=Nullable(1234))
  isnull(seed) || srand(get(seed))

  svr = LinearSVR(C=C)
  cv_gen = (n) -> RandomSub(n, round(Int64, .8 * n), 5)
  learning_curve(svr, cv_gen, m, edge_filter)
end
