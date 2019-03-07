library(flacco)
files = list.files()
features = NULL
extra_features = data.frame(num_samples = integer(), dimensionality = integer(), is_uniform = integer())
extra_feature_names = c('numsamples', 'dimensionality', 'is_uniform')
for (file in files) {
  dat = read.csv(file)
  num_sample = dim(dat)[1]
  num_cols = dim(dat)[2]
  is_uni = grepl('uniform', file)*1
  inputs = dat[1:(num_cols-2)]
  outputs = dat[num_cols]
  inputs = apply(inputs, 2 , as.numeric)
  outputs = apply(outputs, 1, as.numeric)
  feat.object = createFeatureObject(X = inputs, y = outputs)
  new_extra = data.frame(num_sample, num_cols-2, is_uni)
  names(new_extra) = extra_feature_names
  extra_features = rbind(extra_features, new_extra)
  features = rbind(features, data.frame(calculateFeatureSet(feat.object, set = "ela_meta")))
}
files = data.frame(files)
features = cbind(files, features, extra_features)
write.csv(features, file = 'features.csv')
