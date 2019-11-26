- Integral image first
- spend a lot of time on what features to use instead having lots of little-information-features that have to be evaluated with a huge/expensive classifier
- features shapes could be made up
  - (we usually use edge features (2), line features (2), and four-rectangle features (1))



# AdaBoost

- is there a good classifier for each example for each dimension?
  - => find the best separator (represented by a single data point)
  - then reweight all data points (incorreclty classified get heavier)
- Use multi-processing: 1 process for 1 image and all features (`tqdm`, `multiprocessing`)