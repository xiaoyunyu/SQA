import numpy as np
import pickle

def label_embed(k, n, alpha=1/np.sqrt(2), c=1):
	"""
	:param k: label-embedding dim
	:param n: label num
	:param alpha: decay factor for d_min
	:param c: label-embedding norm
	:return: encoded list
	"""
	# Calculate the upper bound of minimum hamming distance via bisection method
	def bisection(k, n):
		target = k-np.log2(n)
		begin = 1
		end = k
		mid = (begin+end)//2
		while end-begin > 1:
			score = 2*(mid-1)-np.log2(mid)-target
			if score == 0:
				return mid
			elif score > 0:
				end = mid
			else:
				begin = mid
			mid = (begin+end)//2
		return mid

	enc_list = []
	# Get upper bound of minimum hamming distance between each pair of labels
	d_min = bisection(k, n)
	# Get decayed minimum hamming distance owning to the unreachable upper-bound
	d_decay = d_min*alpha
	# Generate the first k-dim label-embedding via Bernoulli distribution
	p_i = np.random.binomial(1, 0.5, k)
	enc_list.append(p_i)
	is_valid = True
	for i in range(0, n):
		while is_valid:
			# Generate i-th k-dim label-embedding via Bernoulli distribution
			p_i = np.random.binomial(1, 0.5, k)
			is_valid = False
			# Determine if hamming distance between current i-th embedding and each encoded label
			# is greater than predefined distance. If not, regenerate i-th embedding until satisfy the request.
			for p_j in enc_list:
				if np.sum(p_i ^ p_j) < d_decay:
					is_valid = True
					break
		enc_list.append(p_i)
		is_valid = True
		print('%d th done!'%i)
	# Normalize encoded embedding so that the norm of embedding equals to predefined constant c.
	for i in range(0, n):
		enc_list[i] = (2*enc_list[i]-1)*np.sqrt(c**2/k)
	return enc_list

lis = label_embed(300,7523)

pickle.dump(lis,open('relEmbed.pkl','wb'))
file=open('relEmbed.txt','w')
for num in range(0,7523):
	arr = lis[num].tolist()
	arr = [str(x) for x in arr]
	file.write(' '.join(arr)+'\n')
print('saving relEmbed.txt')
file.close()