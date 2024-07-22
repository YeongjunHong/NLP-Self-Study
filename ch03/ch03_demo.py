"""
밑바닥부터 시작하는 딥러닝 2 , ch03 word2vec
"""

"""
단어의 분산 표현/ 통계기반이 아닌 추론 기반 

word2vec 중에서  CBOW --> Continuous Bag of words  
"""

"""
추론 기반 학습은 학습 데이터의 일부를 사용하여 순차적 학습 
"""


"""
CBOW의 추론 방식

You ?? goodbye and I say hello
"""

import numpy as np
import sys
from layers import MatMul
from utils import preprocess, create_context_target, convert_one_hot
sys.path.append("..")

# 샘플 맥락 데이터 

c0 = np.array([[1,0,0,0,0,0,0]])
c1 = np.array([[0,0,1,0,0,0,0]])

# 가중치 초기화 

W_in = np.random.randn(7,3)
W_out = np.random.randn(3,7)

# 계층생성 

in_layer_0 = MatMul(W_in)
in_layer_1 = MatMul(W_in)
out_layer= MatMul(W_out)

#순전파 

h0 = in_layer_0.forward(c0)
h1 = in_layer_1.forward(c1)
h = 0.5*(h0+h1)

s = out_layer.forward(h)

# print(s)

text = "You say goodbye and I say hello."

corpus, word_to_id, id_to_word = preprocess(text)
# print(corpus)
# print(id_to_word)

contexts, target  = create_context_target(corpus, window_size =1 )

vocab_size = len(word_to_id)
target = convert_one_hot(target,vocab_size)
target = convert_one_hot(contexts,vocab_size)