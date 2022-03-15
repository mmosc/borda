import numpy as np


def ndcg(ranked_list, pos_items, relevance=None, at=None):

  ''' Compute NDCG score, based on https://github.com/recsyspolimi/RecSys_Course_AT_PoliMi implementation. '''

  if relevance is None:
      relevance = np.ones_like(pos_items, dtype=np.int32)
  assert len(relevance) == pos_items.shape[0]

  # Create a dictionary associating item_id to its relevance
  # it2rel[item] -> relevance[item]
  it2rel = {it: r for it, r in zip(pos_items, relevance)}

  # Creates array of length "at" with the relevance associated to the item in that position
  rank_scores = np.asarray([it2rel.get(it, 0.0) for it in ranked_list[:at]], dtype=np.float32)
  # IDCG has all relevances to 1, up to the number of items in the test set
  ideal_dcg = dcg(np.sort(relevance)[::-1])
  # DCG uses the relevance of the recommended items
  rank_dcg = dcg(rank_scores)
  if rank_dcg == 0.0:
      return 0.0

  ndcg_ = rank_dcg / ideal_dcg

  return ndcg_


def dcg(scores):

  ''' Compute DCG score, based on https://github.com/recsyspolimi/RecSys_Course_AT_PoliMi implementation. '''

  return np.sum(np.divide(np.power(2, scores) - 1, np.log(np.arange(scores.shape[0], dtype=np.float32) + 2)),
                  dtype=np.float32)


def recall(ranked_list, pos_items):

  ''' Compute Recall, based on https://github.com/recsyspolimi/RecSys_Course_AT_PoliMi implementation. '''

  is_relevant = np.in1d(ranked_list, pos_items, assume_unique=True)
  recall_score = np.sum(is_relevant, dtype=np.float32) / pos_items.shape[0]

  assert 0 <= recall_score <= 1, recall_score
  return recall_score


def precision(ranked_list, pos_items):

  ''' Compute Precision, based on https://github.com/recsyspolimi/RecSys_Course_AT_PoliMi implementation. '''

  is_relevant = np.in1d(ranked_list, pos_items, assume_unique=True)
  if len(is_relevant) == 0:
      precision_score = 0.0
  else:
      precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

  assert 0 <= precision_score <= 1, precision_score
  return precision_score


def average_precision(ranked_list, pos_items):

  ''' Computes MAP, based on https://github.com/recsyspolimi/RecSys_Course_AT_PoliMi implementation. '''

  is_relevant = np.in1d(ranked_list, pos_items, assume_unique=True)
  if len(is_relevant) == 0:
      a_p = 0.0
  else:
      p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
      a_p = np.sum(p_at_k) / np.min([pos_items.shape[0], is_relevant.shape[0]])

  assert 0 <= a_p <= 1, a_p
  return a_p