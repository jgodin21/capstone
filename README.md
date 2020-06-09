# CAPSTONE PROJECT: SEGMENTING THE OPEN SOURCE HIT BY PITCH DATABASE

## Data Science Problem

The Open Source Hit By Pitch Database was conceived by Rob Mains of *Baseball Prospectus*, and opened to the world in an article on April 14, 2020. [Veteran Presence: The Open Source Hit By Pitch Database](https://www.baseballprospectus.com/news/article/58297/veteran-presence-the-open-source-hit-by-pitch-database/?utm_source=Baseball+Prospectus&utm_campaign=37494e2b6c-2020-04-14+Baseball+Prospectus+Premium+Newsletter&utm_medium=email&utm_term=0_1613350d6d-37494e2b6c-369464685). The database consists of every hit-by-pitch event in Major League Baseball since 1969 (there are over 62,000 of them!).

In Rob's words:
> The point here, though, isn’t for me to show you some numbers. It’s for you to generate some of your own. I’m a guy whose programming skills don’t extend beyond nested IFs and VLOOKUPs in Excel. I know many of you can do much better. So we’re opening up our spreadsheet, all 1.677 million cells of it to you. Download it. See what you can make of it.

As a data scientist and avid sports fan, this dataset appeals to me. It's a quirky side of baseball that should be fun to delve into, and its size lends itself to some interesting analysis, exploratory like Rob started, but perhaps there's something more here. With baseball on an indefinite hiatus, this project will help me stay engaged with the sport as well.

My plan is to use this database as the source for some advanced machine learning, starting with **unsupervised learning** to see whether certain underlying "HBP archetypes" exist - **are there common characteristics that lend themselves to creating groups or types of HBP events?** Do these typologies help tell a more interesting story?

As there are three "scoresheet epochs" in the dataset, it will be necessary to analyze the data using a pattern submodel method due to the systematic missingness of HBP event information in each of the epochs. Those epochs are:

|Scoresheet Epoch|Years|# of Events|<p align="left">Data</p>|
|---|---|---|---|
|Old School Scoresheets|1969-1987|14,101|<p align="left">No balls & strikes nor pitch types/velocities recorded</p>|
|Project Scoresheet|1988-2007|28,056|<p align="left">Balls & strikes recorded, but not pitch types/velocities</p>|
|PITCHf/x Years|2008-present|19,969|<p align="left">All pitch information recorded (though perhaps not perfectly accurately, especially in the early years of this period)</p>|

While the Project Scoresheet era actually kicked off in 1984, the new balls and strikes data doesn't appear in the database until 1988. It's still the best initial moniker for the era that I could come up with.

The plan is to start with segmenting the PITCHf/x Epoch data, and then work backwards. I can then try to see if there are common segment characteristics that exist in each era, even if some data is missing. I'm not sure whether I'll have time to incorporate this into this capstone project, but my ultimate goal is to **create a HBP Typing Tool so new HBP events can be classified on the fly** - the basis of this classification will be a machine learning classifier (most likely using the Xgboost Classifier model).

## Executive Summary

For this project, I combined data from The Open Source Hit By Pitch Database from *Baseball Prospectus*, along with additional data scraped from Stathead.com's Team Batting Event Finder [click here](https://stathead.com/tiny/ar0Pm), plus a custom database of historical MLB franchise name, division, and league changes gleaned from [howtheyplay.com](https://howtheyplay.com/team-sports/major-league-baseball-expansion-and-franchise-relocation), finally adding a number of feature-engineered variables to create a more exhaustive collection of Hit By Pitch data covering the years 1969-2019. Again, the goal is to see whether using advanced unsupervised machine learning tools can help uncover a typology or segmentation of Hit By Pitch events from this data.

After considerable data cleaning both pre- and post-merging of the various datasets, I proceeded by splitting the full dataset into the three Epochs mentioned above, and focused up to this point of the project on analyzing the PITCHf/x Era's data (2008-2019), which is the most complete.

The resulting dataset produced a total of 114 potential segmentation variables after binary coding of categorical variables, including a number of continuous features. Due to the presence of mixed data types and the inability of most standard scikit-learn models to deal with them, I needed to delve into and attempt some newer or less common unsupervised learners.

Ultimately, the unsupervised learning approaches I attempted included **HDBSCAN with Heterogeneous Euclidean-Overlap Metric (HEOM) distances**, and **k-prototypes clustering** with its built-in weighted dissimilarity metric that combines Euclidean distances for continuous data and the count of differences across the categorical features, thus enabling the use of mixed data which is not possible in k-means clustering.

Unfortunately, I was not able to glean interesting results from the HDBSCAN model, which just couldn't find enough density in the data to produce a meaningful set of non-noise clusters. However, k-prototypes did produce some promising clustering solutions, especially after dropping a small number of cases (287) from Epoch 3 that still were missing pitch velocity data. Based on a combination of analyzing a plot of the model costs and silhouette scores for solutions with numbers of clusters ranging from 2 to 14, along with less scientific/more artistic considerations such as seeking good balance among the resulting segment sizes and the ability to tell a good, meaningful story from the segment profiles, an **8-cluster k-prototypes solution was ultimately selected.**

The 8-clusters uncovered by this solution were given the following names, based on careful review of the cluster profiles:

- Cluster 0: Early Plunks
- Cluster 1: Dangerous, But Not Damaging
- Cluster 2: Ejections!
- Cluster 3: Frustration/Revenge Plunks
- Cluster 4: Many Outs & Early Counts
- Cluster 5: Breaking Bad
- Cluster 6: Whoops!
- Cluster 7: Ribs for RBIs

Attempts to utilize Principal Components Analysis (PCA) to aid in visualizing the final segments were also not overly successful, as each Principal Component explained only a relatively small amount of variance in the data. (37 components would be required to explain at least 60% of the variance).

Finally, two supervised learning classifiers were used to attempt to build a classification algorithm for scoring new (or possibly historical) data based on this 8-segment solution. A **Linear Discriminant Analysis (LDA) classifier** was able to achieve 85% correct classification (based on 10-fold cross-validation), but failed to classify the Ejections! cluster at a satisfactory level. However, an **Xgboost (stochastic gradient boosting) classifier** achieved outstanding success in predicting cluster membership from this data, with 10-fold cross-validation **accuracy of 96.5%**, with all clusters in the holdout/test data classified at 93% or better.


## Data Dictionary

**Baseball Prospectus Open Source Hit By Pitch Database**

|Field Name|Description|Value Description/Range|N Missing|
|---|---|---|---|
|DATE|Date when HBP event occurred|4/7/1969 - 9/29/2019|0|
|WIN_TEAM|Team that eventually wins the game|Any of the 30 MLB franchises, 3-letter code used|0|
|PIT_ID|Baseball Prospectus Pitcher ID#|Numeric code|0|
|PITCHER|Pitcher's name|4,301 unique pitchers|0|
|PIT_TEAM|Pitcher's team|Any of 30 MLB franchises, 3-letter code used|0|
|THROWS|Which arm pitcher threw with for the at bat|Left or Right (or Switch)|0|
|BAT_ID|Baseball Prospetus Batter ID#|Numeric code|0|
|BATTER|Batter's name|3,969 unique batters|0|
|BAT_TEAM|Batter's team|Any of 30 MLB franchises, 3-letter code used|0|
|BATS|Stance of the batter for the at bat|Left or Right|0|
|POSITION|Batter's field position played at time of at bat|Numeric position code 1 to 11 (10=DH, 11=PH)|0|
|LINEUP_SPOT|The batting order position of the batter|1 thru 9|0|
|BAT_TEAM_SCORE|Number of runs already scored by batting team at time of HBP|0 to 25 runs|0|
|FLD_TEAM_SCORE|Number of runs already scored by pitching team at time of HBP|0 to 21|0|
|BASES_STAT|Number of baserunners on base (and on which base) at time of at bat|12 values, from ___ to 123 (empty to full)|0|
|HALF|Half inning at time of HBP|Top or Bottom|0|
|INNING|Inning at time of HBP|1 to 22|0|
|OUTS_CT|Number of outs at time of HBP|0, 1, or 2|0|
|BALLS|Number of called balls at time of HBP|0, 1, 2, or 3|14,104|
|STRIKES|Number of called strikes at time of HBP|0, 1, or 2|14,104|
|PITCH_TYPE|Type of pitch thrown that hit batter|10 pitch types, two-letter codes used|42,447|
|VELOCITY|Velocity of pitch that hit batter|46.6 mph to 104.4 mph|42,447|
|PIT_REMOVED|Whether the pitcher was removed from the game following HBP|Y or N|0|
|BAT_REMOVED|Whether the batter was removed from the game following HBP|Y or N|0|
|PIT_EJECTED|Whether the pitcher was ejected from the game following HBP|Y or N|0|
|BAT_EJECTED|Whether the batter was ejected from the game following HBP|Y or N|0|
|UMPIRE|Name of umpire during at bat when HBP occurred|355 unique umpires|1,124|
|EJECTIONS_CT|Total number of ejections in game as a result of HBP (may include other players and/or coaches, not just batter or pitcher)|0 to 9|0|
|EJECTIONS_PL|Player names/IDs for all players ejected due to the HBP|Numeric codes|61,713|
|EJECTION_DES|Text description of what led to ejections|Text|61,713|

**Data Scraped from Stathead.com's Baseball Event Finder**

|Field Name|Description|Value Description/Range|N Missing|
|---|---|---|---|
|EVENT_ID|Event # for teh range of data selected (originally labeled Yr# on Stathead)|1 to 62,129|0|
|GAME_STAT|Number of HBP events occuring in game previous to and including this at bat (originally labeled Game# on Stathead)|1 to 7|0|
|DATE|Date of HBP event|4/7/1969 to 9/29/2019|0|
|BAT_ID|Baseball Reference ID# for batter|Alphanumeric code|0|
|BATTER|Batter's name|3,969 unique batters|0|
|BAT_TEAM|Batter's team|Any of 30 MLB franchises, 3-letter code used|0|
|PIT_ID|Baseball Reference ID# for pitcher|Alphanumeric code|0|
|PITCHER|Pitcher's name|4,301 unique pitchers|0|
|PIT_TEAM| Pitcher's team|Any of 30 MLB franchises, 3-letter code used|0|
|HALF_INNING|Inning (including half) when HBP occurred|t1 to b22 (t=Top, b=Bottom)|0|
|ON_BASE|Number of baserunners on base (and on which base) at time of at bat|12 values, from ___ to 123 (empty to full)|0|
|OUTS|Number of outs at time of HBP|0, 1, or 2|0|
|PIT_COUNT|Number of pitches thrown by pitcher in this at bat|Includes the number of pitches as well as the count itself [e.g. 11 (3-2)]|14,429|
|RBI_ON_PLAY|Whether a run was driven in by batter due to the HBP|0 (No) or 1 (Yes)|0|
|WIN_PROB_ADDED|Given average teams, the change in win probability caused by this batter during the game|-1 to 1|0|
|BASE_OUT_RUNS_ADDED|Given the bases occupied/number of outs situation, how many runs did the batter add in the resulting play. Normalized to per 24 out bases (aka RE24)|0 is average, above 0 is better than average|0|
|LEVERAGE_INDEX|The pressure the pitcher or batter faced during this at bat|1 is average, below 1 is low, above 1 is high|0|

## Contents
**Code**
- [Technical Notebook](.code/capstone-technical-notebook.ipynb)

**Tableau**
- [Tableau Workbook](.tableau/hbp-database.twb)
- [Saved Tableau Visualization Images](.tableau/)

**Report**
- [Capstone Presentation](./report/Typing%20the%20Open%20Source%20HBP%20Database%20-%20Presentation%20version.pdf)


## Conclusions and Next Steps

Creating a typology of Hit By Pitch events for the PITCHf/x years proved challenging, but ultimately I was quite pleased with the final 8-cluster solution uncovered using k-prototypes clustering. The cluster sizes are relatively well-balanced, and a meaningful and differentiable story emerges from the cluster profiles. And, they seem to be strongly predictable using an Xgboost classifier.

That said, a lot of work is left to be done, including but not limited to:
- Clustering the data from Epochs 1 & 2
  - This can be accomplished in potentially two different ways:
    - Use the existing Xgboost classification algorithm to classify Epochs 1 & 2 using the typology built from Epoch 3 data
      - While Xgboost can handle missing data, the resulting segment sizes may be distorted, especially for Epoch 1 which is missing the most data
    - Continue using a pattern sub-model method, where a unique typology is created for each Epoch individually using unsupervised learning
      - Resulting segments/types can be compared across eras to see whether any overlap exists
- Another approach could include continuing to work with the Epoch 3 data, but tuning the weight parameter of the k-prototypes algorithm to skew towards relying more on the categorical/binary data for uncovering HBP types so that historical classifications might work a little better
  - The 287 cases with missing data from Epoch 3 might actually fit better with Epoch 2 data, which shares the same pattern of missingness, so that could be another consideration
- Continue to refine the Xgboost model to reduce any remaining overfitting and/or produce a more compact model
- Build a Flask app to score new (or historical) data into one of the types from this report or from further analysis above
- Lots and lots of additional visualizations could also be created – there’s so much data to play with here!
