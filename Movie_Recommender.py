# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 02:00:56 2022

@author: yunus
"""

import pandas as pd

import numpy as np

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics.pairwise import pairwise_distances

import pickle 

import tensorflow as tf

import os



gc_s_ENCODER_FOLDER_PATH = 'Encoders'
gc_s_MODEL_FOLDER_PATH = 'Models'


class SourceData():
    
    gc_a_GENRES = [
        "Action",
        "Adventure", 
    	"Animation", 
    	"Children's", 
    	"Comedy", 
    	"Crime", 
    	"Documentary", 
    	"Drama", 
    	"Fantasy", 
    	"Film-Noir", 
    	"Horror", 
    	"Musical", 
    	"Mystery", 
    	"Romance", 
    	"Sci-Fi", 
    	"Thriller", 
    	"War",
    	"Western"
    ]
        
    

    def dfGetUsers():
        aColNamesUsers = ['user_id', 'gender', 'age', 'occupation','zip_code']
        dfUsers = pd.read_csv(r'Data\users.dat', delimiter = '::', names =aColNamesUsers , engine = 'python', encoding='latin-1', index_col = 'user_id')
        
        return dfUsers


    def dfGetMovies():
        aColNamesMovies = ['movie_id', 'title', 'genres']
        dfMovies = pd.read_csv(r'Data\movies.dat', delimiter = '::', names =aColNamesMovies , engine = 'python', encoding='latin-1', index_col = 'movie_id')
        
        return dfMovies 
    
    
    def dfGetRatings():
        aColNamesRatings = ['user_id', 'movie_id', 'rating' ,'timestamp']
        dfRatings = pd.read_csv(r'Data\ratings.dat', delimiter = '::', names =aColNamesRatings , engine = 'python')
        return dfRatings
    
    
class PreProcessing():
    
    
    
    def dfExtractYearOfReleaseOfMovies(dfMovies):
        
        dfMovies['year_of_release'] = dfMovies['title'].str[-5:-1].astype(int)
        dfMovies['title'] = dfMovies['title'].str[:-7]
        
        return dfMovies
    
    
    def dfTransformGenresToOneHot(dfToTransform):
        gc_a_GENRES = SourceData.gc_a_GENRES
        for sGenre in gc_a_GENRES:
            dfToTransform.loc[:, sGenre] = dfToTransform.loc[: ,'genres'].str.contains(sGenre).astype(int)
            
        return dfToTransform
        
        
        
    def dfMergeRatingsMoviesUsers(p_dfRatings, p_dfMovies, p_dfUsers):
        dfToReturn= pd.merge(left = p_dfRatings, right= p_dfMovies, left_on = 'movie_id', right_index = True, how = 'left')
        dfToReturn= pd.merge(left = dfToReturn, right= p_dfUsers, left_on = 'user_id', right_index = True, how = 'left')
    
        return dfToReturn
    
    
    
    def oEncodeFieldWithOrdinalEncoder(dfToEncode, sFieldToEncode):
        oEncoder = OrdinalEncoder()
        dfToEncode['{}_encoded'.format(sFieldToEncode)] =  oEncoder.fit_transform(dfToEncode[sFieldToEncode].values.reshape(-1 , 1))
        
        sFilePathToSave = os.path.join(gc_s_ENCODER_FOLDER_PATH ,'{}_encoder.sav'.format(sFieldToEncode))
        
        pickle.dump(oEncoder, open(sFilePathToSave, 'wb'))
        
        return dfToEncode
    
    
    
    def dfGetPreprocessedData():
        dfUsers = SourceData.dfGetUsers()
        dfMovies = SourceData.dfGetMovies()
        dfRatings = SourceData.dfGetRatings()
        
        dfMovies = PreProcessing.dfExtractYearOfReleaseOfMovies(dfMovies)
        
        dfMovies = PreProcessing.dfTransformGenresToOneHot(dfMovies)
        
        dfPreprocessed = PreProcessing.dfMergeRatingsMoviesUsers(dfRatings, dfMovies, dfUsers)
        
        dfPreprocessed = PreProcessing.oEncodeFieldWithOrdinalEncoder(dfPreprocessed, 'movie_id')
        dfPreprocessed = PreProcessing.oEncodeFieldWithOrdinalEncoder(dfPreprocessed, 'user_id')


        return dfPreprocessed
    
    
class DeepLearning():
    
    c_s_COL_NAME_MOVIE_ID_ENCODED = 'movie_id_encoded'
    c_s_COL_NAME_USER_ID_ENCODED = 'user_id_encoded'
    
    
    aColsMovie_X =  [ c_s_COL_NAME_MOVIE_ID_ENCODED] + SourceData.gc_a_GENRES
    aColsUser_X =[c_s_COL_NAME_USER_ID_ENCODED] + SourceData.gc_a_GENRES
    aCol_y = ['rating']
    
    
    gc_s_RATING_ENCODER_NAME = os.path.join(gc_s_ENCODER_FOLDER_PATH ,'rating_encoder.sav')
    gc_s_RATING_ESTIMATOR_MODEL_PATH = os.path.join(gc_s_MODEL_FOLDER_PATH , "rating_estimator_model")
    
    c_i_EMBEDDING_SIZE = 5
    
    def ixSplitTrainValidationTest(dfToSplit):
        
        c_fTrainingRatio = 0.7
        c_fValidationRatio = 0.15
        c_fTestRatio = 0.15
        
        ixTrain,ixTest = train_test_split(
            dfToSplit.index,
            test_size=1-c_fTrainingRatio,
            shuffle=True)
        
        ixValidation,ixTest= train_test_split(
            ixTest,
            test_size=c_fTestRatio/(c_fTestRatio + c_fValidationRatio),
            shuffle=True)
    
        
        return ixTrain, ixValidation, ixTest
    
    def aBuildInputDatasets(df,ixTrain, ixValidation, ixTest):
        aMovie_X_Train = df.loc[ixTrain , DeepLearning.aColsMovie_X].values
        aMovie_X_Validation = df.loc[ ixValidation , DeepLearning.aColsMovie_X].values
        aMovie_X_Test = df.loc[ ixTest , DeepLearning.aColsMovie_X].values
        
        
        aUser_X_Train = df.loc[ixTrain , DeepLearning.aColsUser_X].values
        aUser_X_Validation = df.loc[ ixValidation , DeepLearning.aColsUser_X].values
        aUser_X_Test = df.loc[ ixTest , DeepLearning.aColsUser_X]
        
        
        return aMovie_X_Train, aMovie_X_Validation, aMovie_X_Test, aUser_X_Train, aUser_X_Validation, aUser_X_Test
    
    
    def EncodeOutputOneHotEncoding(df):
        oOheRating = OneHotEncoder()
        oOheRating.fit(df[DeepLearning.aCol_y].values.reshape(-1,1))
        
        pickle.dump(oOheRating, open(DeepLearning.gc_s_RATING_ENCODER_NAME, 'wb'))
        
        
        
        
    def aBuildOutputDatasets(df, ixTrain, ixValidation, ixTest):
        
        DeepLearning.EncodeOutputOneHotEncoding(df)
        
        oOheRating = pickle.load(open(DeepLearning.gc_s_RATING_ENCODER_NAME, 'rb'))
        
        a_y_Train =  oOheRating.transform(df.loc[ ixTrain , DeepLearning.aCol_y].values.reshape(-1,1)).toarray()
        a_y_Validation = oOheRating.transform(df.loc[ ixValidation , DeepLearning.aCol_y].values.reshape(-1,1)).toarray()
        a_y_Test = oOheRating.transform(df.loc[ ixTest , DeepLearning.aCol_y].values.reshape(-1,1)).toarray()
        
        
        return a_y_Train, a_y_Validation, a_y_Test
    
    
    def iGetNrOfUniqueMoviesUsers(df):
        iNrOfMovies = df[DeepLearning.c_s_COL_NAME_MOVIE_ID_ENCODED].nunique()
        iNrOfUsers = df[DeepLearning.c_s_COL_NAME_USER_ID_ENCODED].nunique()
        
        
        return iNrOfMovies, iNrOfUsers
    
    
    def oBuildModel(iNrOfMovies, iNrOfUsers):
        
        aMovieInput = tf.keras.Input(
            shape = (len(DeepLearning.aColsMovie_X),),
            name = 'movie_input'
        )
        
        aMovieEmbedding = tf.keras.layers.Embedding(
            input_dim=iNrOfMovies, # size of vocabulary
            output_dim=DeepLearning.c_i_EMBEDDING_SIZE, # length of sequence
            name = 'movie_embedding'
        )(aMovieInput)
        
        
        
        aUserInput = tf.keras.Input(
            shape = (len(DeepLearning.aColsUser_X),),
            name = 'user_input'
        )
        
        aUserEmbedding = tf.keras.layers.Embedding(
            input_dim=iNrOfUsers, # size of vocabulary
            output_dim=DeepLearning.c_i_EMBEDDING_SIZE, # length of sequence
            name = 'user_embedding'
        )(aUserInput)
    
    
    
        aOutput = tf.keras.layers.Dot(
            name = 'dot_product', 
            normalize = True,
            axes = 2)([aMovieEmbedding, aUserEmbedding])
        
        aOutput = tf.keras.layers.Flatten()(aOutput)
        aOutput = tf.keras.layers.Dense(40, activation = 'ReLU')(aOutput) #, activity_regularizer = tf.keras.regularizers.L2(0.01)
        aOutput = tf.keras.layers.Dropout(0.1)(aOutput)
        aOutput = tf.keras.layers.Dense(20, activation = 'ReLU')(aOutput) #, activity_regularizer = tf.keras.regularizers.L2(0.01)
        aOutput = tf.keras.layers.Dropout(0.1)(aOutput)
        aOutput = tf.keras.layers.Dense(10, activation = 'ReLU')(aOutput) #, activity_regularizer = tf.keras.regularizers.L2(0.01)
        aOutput = tf.keras.layers.Dropout(0.1)(aOutput)
        aOutput = tf.keras.layers.Dense(5, activation = 'sigmoid')(aOutput)
        
        
        oModelRatingEstimator = tf.keras.Model(inputs=[aMovieInput, aUserInput], outputs=aOutput)

        oOptimizer = tf.keras.optimizers.Adam(learning_rate=1e-03)
        
        oModelRatingEstimator.compile(optimizer=oOptimizer,loss= tf.keras.losses.BinaryCrossentropy())
        
        
        return oModelRatingEstimator
    
    
    
    def dfFitModel(oModelRatingEstimator, aMovie_X_Train, aUser_X_Train,a_y_Train, aMovie_X_Validation, aUser_X_Validation, a_y_Validation):
        oEarlyStop = tf.keras.callbacks.EarlyStopping(
            monitor = 'val_loss', 
            mode = 'min', 
            verbose = 0 , 
            patience = 10, 
            restore_best_weights = True)
        
        oModelRatingEstimator.fit(
            [aMovie_X_Train, aUser_X_Train], 
            a_y_Train, 
            epochs= 1000, 
            batch_size=2**12, 
            verbose=1, 
            validation_data= ([aMovie_X_Validation, aUser_X_Validation], a_y_Validation),
            callbacks=[oEarlyStop]
        )
        
        dfHistory = pd.DataFrame(oModelRatingEstimator.history.history)
        
        oModelRatingEstimator.save(DeepLearning.gc_s_RATING_ESTIMATOR_MODEL_PATH)
        
        return dfHistory
    

    def aDecodeOutput(aToDecode):
        oOheRating = pickle.load(open(DeepLearning.gc_s_RATING_ENCODER_NAME, 'rb'))
        aToReturn = oOheRating.inverse_transform(aToDecode)
        
        return aToReturn
    

    
    def aPredict(aMovie_X_Test, aUser_X_Test, oModelRatingEstimator ):
        
        a_y_pred = oModelRatingEstimator.predict([aMovie_X_Test, aUser_X_Test])
        
        a_y_pred = DeepLearning.aDecodeOutput(a_y_pred)
        
        return a_y_pred
    
    
    def dfGetRegressionMetrics(ground_true, pred):
        fMse =  round(mean_squared_error(ground_true, pred), 2)
        fRmse = round(np.sqrt(fMse),2)
        fMae = round(mean_absolute_error(ground_true, pred),2)
        fR2 =  round(r2_score(ground_true, pred),2)
        
        
        dicMetrics = {
            'Mean Squared Error':fMse,
            'Root Mean Squared Error':fRmse,
            'Mean Absolute Error':fMae,
            'R2 Score':fR2
            }
        
        dfRegressionMetrics = pd.DataFrame.from_dict(
            data = dicMetrics,
            orient = 'index',
            columns = ['Value']
            )
        
        return dfRegressionMetrics
    
    
    def oGetEstimatorModel():
        oRatingEstimatorModel = tf.keras.models.load_model(DeepLearning.gc_s_RATING_ESTIMATOR_MODEL_PATH)
        return oRatingEstimatorModel
    

class Recommendation():
    
    c_i_NR_OF_MOST_SIMILAR_USERS = 3
    c_i_TOP_N_ITEMS_TO_RECOMMEND = 5
    
    def __init__(self, p_iUserId, p_dfPreprocessed, p_oModelRatingEstimator = None):
        
        self.iUserId = p_iUserId
        self.dfPreprocessed = p_dfPreprocessed
        
        self.GetEncodersUserAndMovieIds()
        self.GetSimilarityMatrix()
        self.GetEncodedUserAndMovieIds()

        self.iUserIdEncoded = int(self.oUserIdEncoder.transform(np.array(self.iUserId).reshape(-1, 1))[0][0])
        if p_oModelRatingEstimator == None:
            self.oModelRatingEstimator = tf.keras.models.load_model(DeepLearning.gc_s_RATING_ESTIMATOR_MODEL_PATH)
        else:
            self.oModelRatingEstimator = p_oModelRatingEstimator
        
        
    
        
        
    def GetEncodersUserAndMovieIds(self):
        self.oUserIdEncoder = pickle.load(open(os.path.join(gc_s_ENCODER_FOLDER_PATH ,'user_id_encoder.sav'), 'rb'))
        self.oMovieIdEncoder = pickle.load(open(os.path.join(gc_s_ENCODER_FOLDER_PATH ,'movie_id_encoder.sav'), 'rb'))  
        
        
       
    def GetEncodedUserAndMovieIds(self):
        
        aEncodedUserIds = self.oUserIdEncoder.transform(self.oUserIdEncoder.categories_[0].reshape(-1, 1))
        self.aEncodedUserIds = np.sort(aEncodedUserIds)
        
        
        aEncodedMovieIds = self.oMovieIdEncoder.transform(self.oMovieIdEncoder.categories_[0].reshape(-1, 1))
        self.aEncodedMovieIds = np.sort(aEncodedMovieIds)     
        

    def GetSimilarityMatrix(self):
        oModelRatingEstimator = tf.keras.models.load_model(DeepLearning.gc_s_RATING_ESTIMATOR_MODEL_PATH)
        
        aLatentFeaturesUsers = oModelRatingEstimator.get_layer('user_embedding').weights[0]
        aLatentFeaturesMovies= oModelRatingEstimator.get_layer('movie_embedding').weights[0]
        
        
        self.aUserSimilarities =  pairwise_distances(aLatentFeaturesUsers, metric='cosine')
        self.aMovieSimilarities =  pairwise_distances(aLatentFeaturesMovies, metric='cosine')
        
        
        
    def aGetMostSimilarUsers(self):
        aMaxSimilarNUsers = np.argsort(self.aUserSimilarities[self.iUserIdEncoded])
        aMaxSimilarNUsers = aMaxSimilarNUsers[aMaxSimilarNUsers != self.iUserIdEncoded]
        aMaxSimilarNUsers = aMaxSimilarNUsers[0:Recommendation.c_i_NR_OF_MOST_SIMILAR_USERS]
        
        
        return aMaxSimilarNUsers
    
    
    def dfGetMovieIdsRatedByMostSimilarUsers(self, aMaxSimilarNUsers):
        
        dfCosineDistancesToSimilarUsers = pd.DataFrame(
            index = aMaxSimilarNUsers,
            data = self.aUserSimilarities[self.iUserIdEncoded, aMaxSimilarNUsers],
            columns = ['cosine_distance']
        )
        
        dfSimilarUserRatings = self.dfPreprocessed[self.dfPreprocessed['user_id_encoded'].isin(aMaxSimilarNUsers)]
        
        dfSimilarUserRatings = pd.merge(
            left=dfSimilarUserRatings,
            right = dfCosineDistancesToSimilarUsers, 
            left_on = 'user_id_encoded',
            right_index = True
        )
        
        dfSimilarUserRatings['similarity_weighted_rating'] = dfSimilarUserRatings['rating']/dfSimilarUserRatings['cosine_distance']
        dfSimilarUserRatings = dfSimilarUserRatings.groupby(['movie_id_encoded']).sum()['similarity_weighted_rating'].to_frame()
        dfSimilarUserRatings = (dfSimilarUserRatings['similarity_weighted_rating']/(np.sum(1/dfCosineDistancesToSimilarUsers.values))).to_frame()
        
        dfSimilarUserRatings = dfSimilarUserRatings.sort_values(by = 'similarity_weighted_rating',  ascending = False)
        dfSimilarUserRatings.reset_index(inplace = True)
        
        return dfSimilarUserRatings
    
        
    
    def dfExcludeMovieIdsThatAreAlreadyRatedByUser(self, dfSimilarUserRatings):

        aEncodedMovieIdsRatedByUserId = self.dfPreprocessed[self.dfPreprocessed['user_id_encoded'] == self.iUserIdEncoded]['movie_id_encoded'].values
        dfSimilarUserRatings = dfSimilarUserRatings[dfSimilarUserRatings['movie_id_encoded'].isin(aEncodedMovieIdsRatedByUserId) == False]
        
        return dfSimilarUserRatings
    
    
    
    def dfGetExpectedRatingsForSimilarUserRatings(self, dfSimilarUserRatings):
        
        
        dfFiltered = self.dfPreprocessed[self.dfPreprocessed['movie_id_encoded'].isin(dfSimilarUserRatings['movie_id_encoded'].values)].drop_duplicates(subset = ['movie_id_encoded'])
        dfFiltered.sort_values(by = 'movie_id_encoded', inplace = True)
        dfSimilarUserRatings.sort_values(by = 'movie_id_encoded', inplace = True)
        
                                                                                                                                                        
        a_inference_movie_X = dfFiltered.loc[ : , DeepLearning.aColsMovie_X].values
        a_inference_user_X= dfFiltered.loc[ : , DeepLearning.aColsUser_X].values
        
        

        aExpectedRatings =  DeepLearning.aPredict(a_inference_movie_X, a_inference_user_X, self.oModelRatingEstimator )
        dfSimilarUserRatings.loc[:, 'deep_learning_expectation'] = aExpectedRatings
            
        
        return dfSimilarUserRatings
    
    

    def dfCalculateFinalExpectedScores(self, dfSimilarUserRatings):
        dfSimilarUserRatings['normalized_similarity_weighted_rating'] = MinMaxScaler((1, 5)).fit_transform(dfSimilarUserRatings['similarity_weighted_rating'].values.reshape(-1, 1))
        dfSimilarUserRatings['similarity_and_expectation_weighted_rating'] = dfSimilarUserRatings['normalized_similarity_weighted_rating'] + dfSimilarUserRatings['deep_learning_expectation']
        
        dfSimilarUserRatings['final_expected_score'] = MinMaxScaler((1, 5)).fit_transform(dfSimilarUserRatings['similarity_and_expectation_weighted_rating'].values.reshape(-1, 1))
        dfSimilarUserRatings['expected_rating'] = dfSimilarUserRatings['final_expected_score'].apply(np.ceil)
        
        return dfSimilarUserRatings
        
    def dfGetMoviesToRecommend(self, dfSimilarUserRatings):
        dfToRecommend = dfSimilarUserRatings.copy()
        dfToRecommend.sort_values(by = 'final_expected_score', ascending = False , inplace = True)
        dfToRecommend = dfToRecommend.head(Recommendation.c_i_TOP_N_ITEMS_TO_RECOMMEND)
        
        dfMovies = SourceData.dfGetMovies()
        
        dfToRecommend['movie_id'] =  self.oMovieIdEncoder.inverse_transform(dfToRecommend['movie_id_encoded'].to_frame())
        dfToRecommend = pd.merge(left = dfToRecommend, right = dfMovies, left_on = 'movie_id', right_on = 'movie_id')
        
        
        
        dfToRecommend = dfToRecommend[['title', 'genres', 'expected_rating']]
        
        
        return dfToRecommend
        
        
    def dfGetUserHistory(self):
        dfMovies = SourceData.dfGetMovies()
        dfRatings= SourceData.dfGetRatings()
        
        
        dfUserHistory =  dfRatings[dfRatings['user_id'] == self.iUserId]
        dfUserHistory = pd.merge(left = dfUserHistory , right= dfMovies, left_on = 'movie_id', right_on = 'movie_id' )
        dfUserHistory.sort_values(by = 'rating',  ascending = False, inplace = True)
        dfUserHistory = dfUserHistory[['title', 'genres', 'rating']]
        return dfUserHistory


    def dfGetRecommendation(iUserID, dfPreprocessed, oRatingEstimatorModel):

        oRecommender = Recommendation(iUserID, dfPreprocessed, oRatingEstimatorModel)
        
        aMaxSimilarNUsers = oRecommender.aGetMostSimilarUsers()
            
        dfSimilarUserRatings = oRecommender.dfGetMovieIdsRatedByMostSimilarUsers(aMaxSimilarNUsers)
        dfSimilarUserRatings = oRecommender.dfExcludeMovieIdsThatAreAlreadyRatedByUser(dfSimilarUserRatings)
        dfSimilarUserRatings = oRecommender.dfGetExpectedRatingsForSimilarUserRatings(dfSimilarUserRatings)
        dfSimilarUserRatings = oRecommender.dfCalculateFinalExpectedScores(dfSimilarUserRatings)
        
        dfRecommendations = oRecommender.dfGetMoviesToRecommend(dfSimilarUserRatings)
        
        dfUserHistory= oRecommender.dfGetUserHistory()
        
        return dfRecommendations, dfUserHistory
    