import re
import pandas as pd

class SimilarityJoin:
	def __init__(self, data_file1, data_file2):
		self.df1 = pd.read_csv(data_file1)
		self.df2 = pd.read_csv(data_file2)
		  
	def preprocess_df(self, df, cols): 
		"""
			Write your code!
		""" 
		df[cols] = df[cols].fillna('')
		df['joinKey'] = df[cols].apply(' '.join, axis=1)
		df['joinKey'] = df['joinKey'].apply(lambda x: list(
			filter(
				None, 
				list(
					set(
						[x.lower() for x in re.split(r'\W+', x.rstrip('.'))]
					)
				)
			)
		))
		return df
	
	def filtering(self, df1, df2):
		"""
			Write your code!
		"""
		df1_short = df1[['id', 'joinKey']]
		df2_short = df2[['id', 'joinKey']]
		df1_exploded = df1_short.set_index('id') \
								.apply(pd.Series.explode) \
								.reset_index() \
								.set_index('joinKey')
		df2_exploded = df2_short.set_index('id') \
								.apply(pd.Series.explode) \
								.reset_index() \
								.set_index('joinKey')

		common_tokens = df1_exploded.join(df2_exploded, how='inner', lsuffix='1', rsuffix='2')	

		common_tokens_with_count = common_tokens.groupby(['id1', 'id2']) \
										.size() \
										.reset_index(name='r_intersect_s')

		cand_df = common_tokens_with_count.merge(df1_short.rename(columns={'id':'id1', 'joinKey':'joinKey1'}), on='id1') \
									.merge(df2_short.rename(columns={'id':'id2', 'joinKey':'joinKey2'}), on='id2') \
									[['id1', 'joinKey1', 'id2', 'joinKey2']]

		return cand_df
	  
	def verification(self, cand_df, threshold):
		"""
			Write your code!
		"""
		cand_df['jaccard'] = cand_df.apply(
			lambda x: (len(list(set(x[1]).intersection(set(x[3])))) / len(list(set(x[1]).union(set(x[3]))))), 
			axis=1
		)
		result_df = cand_df[cand_df['jaccard'] >= threshold]
		return result_df
		
	def evaluate(self, result, ground_truth):
		"""
			Write your code!
		"""
		identified_matching_pairs = len(result)
		truly_matching_pairs_in_dataset = len(ground_truth)
		truly_matching_pairs = 0

		ground_truth_dict = {}
		for df1_id, df2_id in ground_truth:
			ground_truth_dict[df1_id] = df2_id

		for df1_id, df2_id in result:
			if (df1_id in ground_truth_dict) and (df2_id == ground_truth_dict[df1_id]):
				truly_matching_pairs += 1
		
		precision = truly_matching_pairs / identified_matching_pairs

		recall = truly_matching_pairs / truly_matching_pairs_in_dataset

		fmeasure = (2 * precision * recall) / (precision + recall)

		return (precision, recall, fmeasure)
		
	def jaccard_join(self, cols1, cols2, threshold):
		new_df1 = self.preprocess_df(self.df1, cols1)
		new_df2 = self.preprocess_df(self.df2, cols2)
		print ("Before filtering: %d pairs in total" %(self.df1.shape[0] *self.df2.shape[0])) 
		
		cand_df = self.filtering(new_df1, new_df2)
		print ("After Filtering: %d pairs left" %(cand_df.shape[0]))
		
		result_df = self.verification(cand_df, threshold)
		print ("After Verification: %d similar pairs" %(result_df.shape[0]))
		
		return result_df
	   
		

if __name__ == "__main__":
	er = SimilarityJoin("Amazon_sample.csv", "Google_sample.csv")
	# er = SimilarityJoin("Amazon.csv", "Google.csv")
	amazon_cols = ["title", "manufacturer"]
	google_cols = ["name", "manufacturer"]
	result_df = er.jaccard_join(amazon_cols, google_cols, 0.5)

	result = result_df[['id1', 'id2']].values.tolist()
	ground_truth = pd.read_csv("Amazon_Google_perfectMapping_sample.csv").values.tolist()
	# ground_truth = pd.read_csv("Amazon_Google_perfectMapping.csv").values.tolist()
	print ("(precision, recall, fmeasure) = ", er.evaluate(result, ground_truth))