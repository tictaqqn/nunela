package nunela

import "log"

func TryGetIndex[T Number](tensor Tensor[T], indices ...int) (int, error) {
	if tensor.Rank() != len(indices) {
		return 0, NewErrDifferentRankAndIndices(tensor, indices)
	}
	index := 0
	shapes := tensor.Shape()
	for i := range shapes[1:] {
		index += indices[i]
		index *= shapes[i+1]
	}
	index += indices[len(indices)-1]
	return index, nil
}

func GetIndex[T Number](tensor Tensor[T], indices ...int) int {
	index, err := TryGetIndex(tensor, indices...)
	if err != nil {
		log.Fatalln(err)
	}
	return index
}

func TryAt[T Number](tensor Tensor[T], indices ...int) (T, error) {
	index, err := TryGetIndex(tensor, indices...)
	if err != nil {
		return T(0), err
	}
	return tensor.Ravel()[index], nil
}

func At[T Number](tensor Tensor[T], indices ...int) T {
	value, err := TryAt(tensor, indices...)
	if err != nil {
		log.Fatalln(err)
	}
	return value
}
