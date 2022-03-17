package nunela

import "github.com/vorduin/nune"

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

func GetIndex[T Number](tensor *nune.Tensor[T], indices ...int) int {
	index, err := TryGetIndex(tensor, indices...)
	if err != nil {
		panic(err)
	}
	return index
}

func TryAt[T Number](tensor *nune.Tensor[T], indices ...int) (T, error) {
	index, err := TryGetIndex(tensor, indices...)
	if err != nil {
		return T(0), err
	}
	return tensor.Ravel()[index], nil
}

func At[T Number](tensor *nune.Tensor[T], indices ...int) T {
	value, err := TryAt(tensor, indices...)
	if err != nil {
		panic(err)
	}
	return value
}
