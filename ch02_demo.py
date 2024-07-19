import sys
sys.path.append('..')
import os
import numpy as np
import matplotlib.pyplot as plt

def preprocess(text):
    

    text=text.lower()
    text = text.replace('.',' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word 

text = "You say Goodbye and I say hello"
corpus, word_to_id, id_to_word = preprocess(text)

# print(corpus)
# print("="*30)
# print(word_to_id)
# print("="*30)
# print(id_to_word)
# print("="*30)

"""
각 단어의 맥락에 포함되는 단어의 빈도를 표로 정리
"""

# C= np.array([
#     [0,1,0,0,0,0,0]
#     ,[1,0,1,0,1,1,0]
#     ,[0,1,0,1,0,0,0]
#     ,[0,0,1,0,1,0,0]
#     ,[0,1,0,1,0,0,0]
#     ,[0,1,0,0,0,0,1]
#     ,[0,0,0,0,0,1,0]
# ],dtype=int)

# print(C[0])
# print(C[4])

"""위의 동시 발생 행렬을 자동으로 만들어보자"""

def create_co_matrix(corpus,vocab_size,window_size =1 ):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=int)

    for idx,word_id in enumerate(corpus):
        for i in range(1,window_size+1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0 :
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id]+=i

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id,right_word_id] += 1

    return co_matrix

#인수로 제로벡터가 들어올 수 있으므로 작은 값을 더해줌으로써 '0으로 나누기' 문제를 해결
def cos_similarity(x,y,eps=1e-8):
    nx = x / np.sqrt(np.sum(x**2)+eps)
    ny = y / np.sqrt(np.sum(y**2)+eps)
    return np.dot(nx,ny)

vocab_size = len(word_to_id)
C=create_co_matrix(corpus, vocab_size)

c0 = C[word_to_id['you']]
c1 = C[word_to_id['i']]
print(f'코사인 유사도  {cos_similarity(c0,c1)}')


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    '''유사 단어 검색

    :param query: 쿼리(텍스트)
    :param word_to_id: 단어에서 단어 ID로 변환하는 딕셔너리
    :param id_to_word: 단어 ID에서 단어로 변환하는 딕셔너리
    :param word_matrix: 단어 벡터를 정리한 행렬. 각 행에 해당 단어 벡터가 저장되어 있다고 가정한다.
    :param top: 상위 몇 개까지 출력할 지 지정
    '''
    if query not in word_to_id:
        print('%s(을)를 찾을 수 없습니다.' % query)
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # 코사인 유사도 계산
    vocab_size = len(id_to_word)

    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # 코사인 유사도를 기준으로 내림차순으로 출력
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return

print("\n You와 가장 유사한 단어 5개는 ?")
most_similar('you',word_to_id,id_to_word,C, top=5)

"""
컴퓨터에게 자연어를 이해 시키는 방법 중에 하나인 통계 방법, 

you 와 i는 인칭대명사라서 납득이 되지만, goodbye와 hello가 유사도가 높다는 것은 이해하기 힘들다.
"""


"""
그 이유는 '동시'발생 행렬 때문인데, 위의 예시는 글이 짧지만 실제로 처리하게될 텍스트는 장문이다. 영어의 특성을 고려했을 때 ( 영어 작문시 paraphrase를 하므로) 
동시 발생 행렬을 그대로 사용한다면 자연어를 이해시키고자 하는 우리의 목적은 실패할 공산이 크다.
"""

"""
위의 문제를 해결하기 위해 점별 상호정보량(Pointwise Mutual Information)라는 척도를 사용


x와 y가 동시에 일어날 확률을 x가 일어날 확률과 y가 일어날 확률을 곱한 값으로 나눈다. 자세한건은 

CBOW에서 더 자세히 
"""

def ppmi(C, verbose=False, eps = 1e-8):
    '''PPMI(점별 상호정보량) 생성
    :param C: 동시발생 행렬
    :param verbose: 진행 상황을 출력할지 여부
    :return:
    '''
        
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C,axis=0)
    total = C.shape[0]*C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i,j]*N / (S[j]*S[i])+eps)
            M[i,j] = max(0,pmi)

            if verbose :
                cnt +=1
                if cnt % (total//100+1) ==0:
                    print("%.1f%%완료 " % (100*cnt/total))

    return M

W = ppmi(C)

np.set_printoptions(precision=3)
print('동시발생행렬')
print(C)
print("="*50)
print('ppmi')
print(W)

"""
조금 더 나아지긴 했지만, 문제는 말뭉치가 커질수록 차원이 더 커진다는 것

그래서 차원 감소 Dimensionality Reduction

그 대표적인 방법이 특잇값 분해 SVD Singular Value Decomposition
"""


U, S, V = np.linalg.svd(W)


print('동시발생행렬')
print(C[0])
print("="*50)
print('ppmi')
print(W[0])
print("="*50)
print(U)
print("="*50)
print(U[0, :2]) # SVD


for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id,0],U[word_id,1]))
plt.scatter(U[:,0],U[:,1],alpha=0.5)
# plt.show()

"""
작은 말뭉치말고 어느 정도 큰 말뭉치를 사용해보자 

PTB Penn TreeBank
"""