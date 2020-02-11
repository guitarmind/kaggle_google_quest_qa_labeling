# 30th Sliver Solution - BERT variants with Discovered Post-Processing Magic

## Text Preprocessing

I only joined the competition in the last two weeks, therefore I didn't spend too much time on the preprocessing step, but some simple tricks did improve the score. Assume that the built-in tokenizers from pretrained BERTs could do a good job for further preprocessing.

 - Remove extra white spaces to make text more dense
 - Unescape html entities, like `&lt;`, `&equals;`, `&gt;`, ...
 - Extract plain text from html tags if found (using `get_text()` of BeautifulSoup)

## Custom BERT Models for Ensembles

As a fan of ensemble learning, I always believe that the **model diversity** is the key for success. I trained 5 BERT-based models with slightly different custom layers, but they share the same structure for BERT embeddings: `[CLS] title [SEP] question [SEP]` for question text and `[CLS] answer [SEP]` for the answer text. I only used the last hidden state output embeddings of `[CLS]` token from BERT models to combine with other layers (pooler output performed worse).

 - **Topology 1: Roberta-Base, Xlnet-Base-Cased**

   - 2 BERT embeddings (CLS): q_embed, a_embed
   - 3 categorical embeddings: Concat(cate_embeds) -> Dense(128, relu)
   - 2 separate FC layer paths
      - Concat(q_embed, cate_embeds_dense) -> Dense(256, relu) -> Dense(21, sigmoid)
      - Concat(a_embed, cate_embeds_dense) -> Dense(256, relu) -> Dense(9, sigmoid)

 - **Topology 2: Roberta-Base, Bert-Base-Uncased, Bert-Base-Cased**

   - 2 BERT embeddings (CLS): q_embed, a_embed
   - 2 separate FC layer paths
      - q_embed -> Dense(256, relu) -> Dense(21, sigmoid)
      - a_embed -> Dense(256, relu) -> Dense(9, sigmoid)

I also discovered that splitting questions and answers into two separate fully-connected layer paths works better than mixing both. It makes sense to me as the labeling of classes by the voters may focus on the content of `title+question` and `answer` separately. Categorical embedding layers for `host`, 1st token of `host` and `category` columns contributed the ensemble score.

The learning rate of all models is fixed to `2e-5`, also applied `ReduceLROnPlateau` for LR decay (factor=0.1) and a custom early stopping callback based on validation Spearman score.

The final model is a weighted average of those models with a post processing to optimize ranks.


## Determinism on TensorFlow 2.1

Reproducibility had been an issue for tensorflow/keras, but this repo from Nvidia helped me to control the determinism to a great deal! Now we can get almost the same result in multiple runs using the same random seed.
This gives us a clear view about the relative performance of all experiments, and then we can gradually improve the models by the right setup and approaches.
https://github.com/NVIDIA/tensorflow-determinism

## Post Processing Magic

Lost of people were discussing about what is the actual trick/magic that can boost the Spearman correlation score. I was originally having no clues about it, but after studying the definition of Spearman correlation and the patterns inside the training set labels, I discovered that we could utilize fixed percentiles of label values to approximate to the optimal rank in each class.

I searched from 1 to 100 as the divisor for fixed percentile intervals using out-of-fold prediction from one of the best ensembles. I finally chose 60 as the fixed divisor because it consistently boosted the score on both local CV and public LB (+~0.03-0.05).

The code is very simple, given the unique labels of training set as the distribution samples:
```
y_labels = df_train[output_categories].copy()
y_labels = y_labels.values.flatten()
unique_labels = np.array(sorted(np.unique(y_labels)))
unique_labels

array([0.        , 0.2       , 0.26666667, 0.3       , 0.33333333,
       0.33333333, 0.4       , 0.44444444, 0.46666667, 0.5       ,
       0.53333333, 0.55555556, 0.6       , 0.66666667, 0.66666667,
       0.7       , 0.73333333, 0.77777778, 0.8       , 0.83333333,
       0.86666667, 0.88888889, 0.9       , 0.93333333, 1.        ])
```

I created 60 optimal percentiles:
```
denominator = 60
q = np.arange(0, 101, 100 / denominator)
exp_labels = np.percentile(unique_labels, q)
exp_labels

array([0.        , 0.08      , 0.16      , 0.21333333, 0.24      ,
       0.26666667, 0.28      , 0.29333333, 0.30666667, 0.32      ,
       0.33333333, 0.33333333, 0.33333333, 0.34666667, 0.37333333,
       0.4       , 0.41777778, 0.43555556, 0.44888889, 0.45777778,
       0.46666667, 0.48      , 0.49333333, 0.50666667, 0.52      ,
       0.53333333, 0.54222222, 0.55111111, 0.56444444, 0.58222222,
       0.6       , 0.62666667, 0.65333333, 0.66666667, 0.66666667,
       0.66666667, 0.68      , 0.69333333, 0.70666667, 0.72      ,
       0.73333333, 0.75111111, 0.76888889, 0.78222222, 0.79111111,
       0.8       , 0.81333333, 0.82666667, 0.84      , 0.85333333,
       0.86666667, 0.87555556, 0.88444444, 0.89111111, 0.89555556,
       0.9       , 0.91333333, 0.92666667, 0.94666667, 0.97333333,
       1.        ])
```

And a mapping function to align BERT outputs to the closest percentile value.
```
def optimize_ranks(preds, unique_labels):
    new_preds = np.zeros(preds.shape)
    for i in range(preds.shape[1]):
        interpolate_bins = np.digitize(preds[:, i],
                                       bins=unique_labels,
                                       right=False)
        
        if len(np.unique(interpolate_bins)) == 1:
            # Use original preds
            new_preds[:, i] = preds[:, i]
        else:
            new_preds[:, i] = unique_labels[interpolate_bins]

    return new_preds

weights = [1.0, 1.0, 1.0, 1.0, 1.0]
oof_preds = val_ensemble_preds(all_val_preds, weights)
magic_preds = optimize_ranks(oof_preds, exp_labels)
blend_score = compute_spearmanr(outputs, magic_preds)
```

The Spearman correlation will become NaN if the output column contains 1 unique value, because in this case the standard deviation will be zero and caused divide-by-zero problem (submission error). The trick I used is to use original predictions from BERT models for that column.

Here is a summary table of original scores versus magic-boosted scores:
| Model                  | Local CV without Magic | Local CV with Magic | Public LB with Magic | Private LB with Magic |
|------------------------|-----------------------:|--------------------:|---------------------:|----------------------:|
| Roberta-Base (T1)      | 0.395972               | 0.414739            | 0.43531              | 0.40019               |
| Xlnet-Base-Cased (T1)  | 0.392654               | 0.407847            | 0.42771              | 0.39609               |
| Roberta-Base (T2)      | 0.398664               | 0.422453            | 0.43522              | 0.40242               |
| Bert-Base-Uncased (T2) | 0.389013               | 0.398852            | 0.41844              | 0.39075               |
| Bert-Base-Cased (T2)   | 0.387040               | 0.400199            | 0.42026              | 0.38455               |
| Final Ensemble         | 0.392669               | 0.438232            | 0.44238              | 0.41208               |


## Things That Didn't Work for Me

They produced worse results on both local CV and LBs
- SpatialDropout1D for embeddings and Dense dropouts
- Separate BERT embeddings for title and question
- Batch normalizations for embeddings and dense layers
