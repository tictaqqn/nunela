package nunela

import (
	"github.com/vorduin/nune"
)

func TryRepeat[T Number](tensor *nune.Tensor[T], baseAxes []int, toShape []int) (*nune.Tensor[T], error) {
	if tensor.Rank() != len(baseAxes) {
		return nil, NewErrDifferentRankAndIndices(tensor, baseAxes)
	}
	for i, baseAxis := range baseAxes {
		if baseAxis >= len(toShape) {
			return nil, NewErrInvalidAxisAndShape(baseAxis, toShape)
		}
		if tensor.Shape()[i] != toShape[baseAxis] {
			return nil, NewErrInvalidSizesWithBaseIndex(tensor.Shape(), toShape, baseAxis)
		}
	}
	prods := make([]int, len(toShape)+1)
	prods[len(toShape)] = 1
	for i := len(toShape) - 1; i >= 0; i-- {
		prods[i] = prods[i+1] * toShape[i]
	}
	data := make([]T, prods[0])
	for toIndex := 0; toIndex < prods[0]; toIndex++ {
		baseIndices := make([]int, len(baseAxes))
		for i, baseAxis := range baseAxes {
			baseIndices[i] = (toIndex % prods[baseAxis]) / prods[baseAxis+1]
		}
		data[toIndex] = At(tensor, baseIndices...)
	}
	newTensor := FromBufferWithShape(data, toShape)
	return &newTensor, nil
}

func Repeat[T Number](tensor *nune.Tensor[T], baseAxes []int, toShape []int) *nune.Tensor[T] {
	out, err := TryRepeat(tensor, baseAxes, toShape)
	if err != nil {
		panic(err)
	}
	return out
}
