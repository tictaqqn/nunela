package nunela

import "github.com/vorduin/nune"

// TryGetIndex returns the index of data of tensor at the given indices.
func TryGetIndex[T Number](tensor *nune.Tensor[T], indices ...int) (int, error) {
	if tensor.Rank() != len(indices) {
		return 0, NewErrDifferentRankAndIndices(tensor, indices)
	}
	index := 0
	shape := tensor.Shape()
	for i := range shape[1:] {
		index += indices[i]
		index *= shape[i+1]
	}
	index += indices[len(indices)-1]
	return index, nil
}

// GetIndex returns the index of data of tensor at the given indices.
func GetIndex[T Number](tensor *nune.Tensor[T], indices ...int) int {
	index, err := TryGetIndex(tensor, indices...)
	if err != nil {
		panic(err)
	}
	return index
}

// TryAt returns the value at the given indices.
func TryAt[T Number](tensor *nune.Tensor[T], indices ...int) (T, error) {
	index, err := TryGetIndex(tensor, indices...)
	if err != nil {
		return T(0), err
	}
	return tensor.Ravel()[index], nil
}

// At returns the value at the given indices.
func At[T Number](tensor *nune.Tensor[T], indices ...int) T {
	value, err := TryAt(tensor, indices...)
	if err != nil {
		panic(err)
	}
	return value
}
