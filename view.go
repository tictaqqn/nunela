package nunela

import (
	"github.com/vorduin/nune"
	"github.com/vorduin/slices"
)

// rangePairs indicates whether the index is in range [min, max).
// if rangePairs do not have the item of the given axis, then allows all range.
func TryView[T Number](tensor *nune.Tensor[T], axisPairs map[int][2]int, dropAxis map[int]struct{}) (*nune.Tensor[T], error) {
	for axis, x := range axisPairs {
		size := tensor.Size(axis)
		if axis >= tensor.Rank() || x[0] >= size || x[1] >= size+1 {
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
			index := (i % prods[axis]) / prods[axis+1]
			if index < x[0] || x[1] <= index {
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
		pair, ok := axisPairs[i]
		if !ok {
			shape = append(shape, size)
		} else {
			len := pair[1] - pair[0]
			adding := true
			if len == 1 && dropAxis != nil {
				_, ok := dropAxis[i]
				adding = !ok
			}
			if adding {
				shape = append(shape, len)
			}
		}
	}
	arr := FromBufferWithShape(data, shape)
	return &arr, nil
}

// TODO: Implement TensorView without copying and methods similar to Tensor
func View[T Number](tensor *nune.Tensor[T], axisPairs map[int][2]int, dropAxis map[int]struct{}) *nune.Tensor[T] {
	out, err := TryView(tensor, axisPairs, dropAxis)
	if err != nil {
		panic(err)
	}
	return out
}

func TryViewAssign[T Number](assigned, assigning *nune.Tensor[T], axisPairs map[int][2]int, dropAxis map[int]struct{}) error {
	for axis, x := range axisPairs {
		size := assigned.Size(axis)
		if axis >= assigned.Rank() || x[0] >= size || x[1] >= size+1 {
			return NewErrInappropriateAxisAndAxisNumber(assigned, axis, x)
		}
	}

	prods := make([]int, assigned.Rank()+1)
	prods[assigned.Rank()] = 1
	for i := assigned.Rank() - 1; i >= 0; i-- {
		prods[i] = prods[i+1] * assigned.Shape()[i]
	}
	for i := range assigned.Ravel() {
		shouldAdd := true
		indices := make([]int, 0, len(axisPairs))
		for axis, x := range axisPairs {
			index := (i % prods[axis]) / prods[axis+1]
			if index < x[0] || x[1] <= index {
				shouldAdd = false
				break
			}
			indices = append(indices, index)
		}
		if shouldAdd {
			assigned.Ravel()[i] = At(assigning, indices...)
		}
	}
	return nil
}

func ViewAssign[T Number](assigned, assigning *nune.Tensor[T], axisPairs map[int][2]int, dropAxis map[int]struct{}) {
	err := TryViewAssign(assigned, assigning, axisPairs, dropAxis)
	if err != nil {
		panic(err)
	}
}
