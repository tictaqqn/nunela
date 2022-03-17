package nunela

import (
	"github.com/vorduin/nune"
	"github.com/vorduin/slices"
)

func TryView[T Number](tensor *nune.Tensor[T], axisPairs map[int]int) (*nune.Tensor[T], error) {
	for axis, x := range axisPairs {
		if axis >= tensor.Rank() || x >= tensor.Size(axis) {
			return nil, NewErrInappropriateAxisAndAxisNumber(tensor, axis, x)
		}
	}

	axes := keys(axisPairs)
	data := make([]T, 0, tensor.Numel()/slices.Prod(mapSlice(axes, func(axis int) int { return tensor.Size(axis) })))

	prods := make([]int, tensor.Rank()+1)
	prods[tensor.Rank()] = 1
	for i := tensor.Rank() - 1; i >= 0; i-- {
		prods[i] = prods[i+1] * tensor.Shape()[i]
	}
	for i, val := range tensor.Ravel() {
		shouldAdd := true
		for axis, x := range axisPairs {
			if (i%prods[axis])/prods[axis+1] != x {
				shouldAdd = false
				break
			}
		}
		if shouldAdd {
			data = append(data, val)
		}
	}
	shape := make([]int, 0, tensor.Rank()-1)
	for i, size := range tensor.Shape() {
		if !slices.Contains(axes, i) {
			shape = append(shape, size)
		}
	}
	arr := FromBufferWithShape(data, shape)
	return &arr, nil
}

// TODO: Implement TensorView without copying and methods similar to Tensor
func View[T Number](tensor *nune.Tensor[T], axisPairs map[int]int) *nune.Tensor[T] {
	out, err := TryView(tensor, axisPairs)
	if err != nil {
		panic(err)
	}
	return out
}
