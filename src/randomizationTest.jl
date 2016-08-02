using DataFrames
using Lazy
using StatsBase


function randomizationTest(compare_fn,
  truths::AbstractVector, perms::AbstractVector;
  num_tests=1000)

  concats = [truths perms]
  num_samples = length(concats)
  num_truths = length(truths)
  num_perms = length(perms)

  map(1:num_tests) do i
    sampled_data = @> concats sample(num_samples, replace=false)
    mock_truths = sampled_data[1:num_truths]
    mock_perms = sampled_data[num_truths+1:end]
    compare_fn(mock_truths, mock_perms)
  end
end


function diffInMeans(truths::AbstractVector, perms::AbstractVector;
  num_tests = 1000)

  meanDiff(a1, a2) = mean(a1) - mean(a2)

  actual = meanDiff(truths, perms)
  tests = randomizationTest(meanDiff, truths, perms, num_tests=num_tests)

  p_pos = sum([s <= actual for s in tests])/num_tests

  return p_pos, actual, tests
end


function diffInMeans(real_coefs::DataFrame, perm_coefs::DataFrame;
  num_tests=1000)

  predictors = names(real_coefs)
  @assert predictors == names(perm_coefs)

  p_pos = map(predictors) do p
    rank, _, _ = diffInMeans(real_coefs[p], perm_coefs[p], num_tests=num_tests)
    rank
  end

  @> DataFrame(predictor=predictors, p_pos=p_pos, p_neg=1 .- p_pos) sort(cols=:p_pos)
end


function diffInMeans(coefs::DataFrame;
  num_tests=1000)

  perm_coefs = begin
    num_samples = size(coefs, 1)
    reduce(DataFrame(), names(coefs)) do df, p
      df[p] = zeros(Float64, num_samples)
      df
    end
  end

  diffInMeans(coefs, perm_coefs, num_tests=num_tests)
end
