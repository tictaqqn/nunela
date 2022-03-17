package nunela

import (
	"github.com/vorduin/nune"
	"github.com/vorduin/slices"
)

// TryView extracts a sub-tensor from a tensor with given axisPairs.
// axisPairs indicates whether the index is in range [min, max).
// if axisPairs do not have the item of the given axis, then allows all range.
// Drop axis if dropAxis has an element axis and the difference between min and max is 1.
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
			length := pair[1] - pair[0]
			adding := true
			if length == 1 && dropAxis != nil {
				_, ok := dropAxis[i]
				adding = !ok
			}
			if adding {
				shape = append(shape, length)
			}
		}
	}
	arr := FromBufferWithShape(data, shape)
	return &arr, nil
}

// View extracts a sub-tensor from a tensor with given axisPairs.
// axisPairs indicates whether the index is in range [min, max).
// if axisPairs do not have the item of the given axis, then allows all range.
// Drop axis if dropAxis has an element axis and the difference between min and max is 1.
// TODO: Implement TensorView without copying and methods similar to Tensor
func View[T Number](tensor *nune.Tensor[T], axisPairs map[int][2]int, dropAxis map[int]struct{}) *nune.Tensor[T] {
	out, err := TryView(tensor, axisPairs, dropAxis)
	if err != nil {
		panic(err)
	}
	return out
}

// TryViewAssign extracts a sub-tensor from a tensor with given axisPairs and assigning tensor is assigned to the sub-tensor.
// axisPairs indicates whether the index is in range [min, max).
// if axisPairs do not have the item of the given axis, then allows all range.
// Drop axis if dropAxis has an element axis and the difference between min and max is 1.
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
		indices := make([]int, 0, len(axisPairs)-1)
		for axis := 0; axis < assigning.Rank(); axis++ {
			x, ok := axisPairs[axis]
			index := (i % prods[axis]) / prods[axis+1]
			if !ok {
				indices = append(indices, index)
				continue
			}
			if index < x[0] || x[1] <= index {
				shouldAdd = false
				break
			}
			if _, ok := dropAxis[axis]; !ok || x[1] != x[0]+1 {
				indices = append(indices, index-x[0])
			}
		}
		if shouldAdd {
			assigned.Ravel()[i] = At(assigning, indices...)
		}
	}
	return nil
}

// ViewAssign extracts a sub-tensor from a tensor with given axisPairs and assigning tensor is assigned to the sub-tensor.
// axisPairs indicates whether the index is in range [min, max).
// if axisPairs do not have the item of the given axis, then allows all range.
// Drop axis if dropAxis has an element axis and the difference between min and max is 1.
func ViewAssign[T Number](assigned, assigning *nune.Tensor[T], axisPairs map[int][2]int, dropAxis map[int]struct{}) {
	err := TryViewAssign(assigned, assigning, axisPairs, dropAxis)
	if err != nil {
		panic(err)
	}
}
