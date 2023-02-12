# anomaly_detection.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import json

class AnomalyDetection():

	def get_fill_values(self, length, position) -> str:
		array = [0] * length
		array[position] = 1
		return ','.join([str(i) for i in array])

	def scaleNumInternal(self, df, ind):
		mean = df[ind].astype(float).mean()
		std = df[ind].astype(float).std()
		df['features'] = df['features'] + ',' + df[ind].apply(lambda x : str((float(x)-mean)/std))
		return df

	def cat2NumInternal(self, df, ind):
		labels = df[ind].unique().tolist()
		df['features'] = df['features'] + ',' + df[ind].apply(lambda x: self.get_fill_values(len(labels), labels.index(x)))
		return df

	def operate_data_frame(self, df, indices, logical_function):
		new_df = pd.DataFrame(df['features'].to_list(), columns=np.arange(len(df['features'].iloc[0])))

		new_df['features'] = np.empty(len(new_df), str)

		for ind in range(len(df['features'].iloc[0])):
			if ind in indices:
				new_df = logical_function(new_df, ind)
			else:
				new_df['features'] = new_df['features'] + ',' + new_df[ind].astype(str)

		new_df['features'] = new_df['features'].apply(lambda x: list(filter(None, x.split(','))))
		new_df = new_df[['features']]

		return new_df
		

	def scaleNum(self, df, indices):
		"""
			Write your code!
		"""
		return self.operate_data_frame(df, indices, self.scaleNumInternal)


	def cat2Num(self, df, indices):
		"""
			Write your code!
		"""
		df['features'] = df['features'].apply(lambda x: x.strip('][').split(', '))
		return self.operate_data_frame(df, indices, self.cat2NumInternal)
		

	def cluster_dataset(self, df, num_clusters):
		new_df = pd.concat(
			[pd.DataFrame(df['features'].to_list(), columns=np.arange(len(df['features'].iloc[0]))),
			df],
			axis=1
		)
		kmeans = KMeans(n_clusters=num_clusters, n_init='auto').fit(new_df.drop(columns='features'))
		new_df['cluster'] = kmeans.predict(new_df.drop(columns='features'))
		return new_df

	def detect(self, df, k, t):
		"""
			Write your code!
		"""
		new_df = self.cluster_dataset(df, k)
		
		cluster_summary = new_df.groupby(by='cluster').size().reset_index(name='count').sort_values(by='count', ascending=False)
		n_max = cluster_summary['count'].iloc[0]
		n_min = cluster_summary['count'].iloc[-1]
		denominator = n_max - n_min

		cluster_dict = cluster_summary.set_index('cluster').to_dict()['count']

		new_df['score'] = new_df['cluster'].apply(lambda x: (n_max - cluster_dict[x])/denominator)
		new_df = new_df[new_df['score'] > t][['features', 'score']]
		return new_df
        
    
 
if __name__ == "__main__":
	df = pd.read_csv('logs-features-sample.csv').set_index('id')
	ad = AnomalyDetection()

	df1 = ad.cat2Num(df, [0,1])
	print(df1)

	df2 = ad.scaleNum(df1, [6])
	print(df2)

	df3 = ad.detect(df2, 8, 0.97)
	print(df3)