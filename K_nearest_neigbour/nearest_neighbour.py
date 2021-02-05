from math import sqrt

def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

dataset = [[5.7, 2.8, 4.1, 1.3,1],
	[6.5,3,5.5,1.8,2],
	[6.3, 2.3, 4.4, 1.3, 1],
	[6.4,2.9,4.3,1.3,1],
	[5.6,2.8,4.9,2,2],
	[5.9,3,5.1,1.8,	2],
	[5.4, 3.4, 1.7, 0.2, 0],
	[6.1, 2.8, 4, 1.3, 1],
	[4.9, 2.5, 4.5, 1.7, 2],
	[5.8, 2.6, 1.2, 1.2, 1],
	[7.1, 3, 5.9, 2.1, 2],
	[5.8, 4, 1.2, 0.2, 0]]

predict_data = [1.5,7.3,9.7,2.4]
predict_data_2 = [2.7, 5.6, 1.2,3.8]
number_of_neighbours = 5
prediction = predict_classification(dataset,predict_data_2 , number_of_neighbours)
print('Got %d.' % (prediction))