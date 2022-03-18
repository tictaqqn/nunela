package nunela

import (
	"sync"

	"github.com/vorduin/nune"
	"github.com/vorduin/slices"
)

// TryTensorDot computes tensor multiplation on the given axes.
func TryTensorDot[T Number](tensors []*nune.Tensor[T], axes []int) (*nune.Tensor[T], error) {
	if len(tensors) != len(axes) {
		return nil, NewErrDifferentLen(tensors, axes)
	}
	if len(tensors) <= 1 {
		return nil, NewErrNotEnoughTensorsGiven()
	}
	lenAxis := tensors[0].Size(axes[0])
	for !every(tensors[1:], func(i int, t *nune.Tensor[T]) bool { return t.Size(axes[i+1]) == lenAxis }) {
		return nil, NewErrDifferentShapes(tensors...)
	}
	var shape []int
	for i := range tensors {
		for j, size := range tensors[i].Shape() {
			if j != axes[i] {
				shape = append(shape, size)
			}
		}
	}
	sum := nune.Zeros[T](shape...)
	prod := nune.Ones[T](shape...)
	r := rangeSlice(len(shape))
	for i := 0; i < lenAxis; i++ {
		rankRange := 0
		for j := range prod.Ravel() {
			prod.Ravel()[j] = T(1)
		}
		for j := range tensors {
			viewRank := tensors[j].Rank() - 1
			MulAssign(&prod, Repeat(View(tensors[j], map[int][2]int{axes[j]: {i, i + 1}}, map[int]struct{}{axes[j]: {}}), r[rankRange:rankRange+viewRank], shape))
			rankRange += viewRank
		}
		AddAssign(&sum, &prod)
	}
	return &sum, nil
}

// TensorDot computes tensor multiplation on the given axes.
func TensorDot[T Number](tensors []*nune.Tensor[T], axes []int) *nune.Tensor[T] {
	out, err := TryTensorDot(tensors, axes)
	if err != nil {
		panic(err)
	}
	return out
}

// TryStrassenDot computes matrix multiplation using Strassen algorithm on given axes.
func TryStrassenDot[T Number](a *nune.Tensor[T], b *nune.Tensor[T], aAxis int, bAxis int) (*nune.Tensor[T], error) {
	if a.Rank() <= 1 || b.Rank() <= 1 {
		return TryTensorDot([]*nune.Tensor[T]{a, b}, []int{aAxis, bAxis})
	}
	size := a.Size(aAxis)
	if size != b.Size(bAxis) {
		return nil, NewErrDifferentShapes(a, b)
	}
	aNewShape := make([]int, a.Rank())
	aNewFlag := false
	for i, size := range a.Shape() {
		if size%2 != 0 {
			aNewFlag = true
			aNewShape[i] = size + 1
		} else {
			aNewShape[i] = size
		}
	}
	var aNew *nune.Tensor[T]
	if aNewFlag {
		aNewTmp := nune.Zeros[T](aNewShape...)
		aNew = &aNewTmp
		ViewAssign(aNew, a, makeAxisPairs(a.Shape(), nil), nil)
	} else {
		aNew = a
	}

	bNewShape := make([]int, b.Rank())
	bNewFlag := false
	for i, size := range b.Shape() {
		if size%2 != 0 {
			bNewFlag = true
			bNewShape[i] = size + 1
		} else {
			bNewShape[i] = size
		}
	}
	var bNew *nune.Tensor[T]
	if bNewFlag {
		bNewTmp := nune.Zeros[T](bNewShape...)
		bNew = &bNewTmp
		ViewAssign(bNew, b, makeAxisPairs(b.Shape(), nil), nil)
	} else {
		bNew = b
	}
	mul, err := tryStrassenDot(aNew, bNew, aAxis, bAxis)
	if err != nil {
		return nil, err
	}

	var mulOut *nune.Tensor[T]
	if aNewFlag || bNewFlag {
		var shape []int
		newAxis := 0
		for i, size := range a.Shape() {
			if i == aAxis {
				continue
			}
			shape = append(shape, size)
			newAxis++
		}
		for i, size := range b.Shape() {
			if i == bAxis {
				continue
			}
			shape = append(shape, size)
			newAxis++
		}
		mulOutTmp := nune.Zeros[T](shape...)
		mulOut = &mulOutTmp
		ViewAssign(mulOut, mul, nil, nil)
	} else {
		mulOut = mul
	}
	return mulOut, nil
}

func tryStrassenDot[T Number](a *nune.Tensor[T], b *nune.Tensor[T], aAxis int, bAxis int) (*nune.Tensor[T], error) {

	size := a.Size(aAxis)

	aOtherAx := aAxis - 1
	if aOtherAx == -1 {
		aOtherAx = 1
	}
	bOtherAx := bAxis - 1
	if bOtherAx == -1 {
		bOtherAx = 1
	}
	aOtherAxSize := a.Size(aOtherAx)
	bOtherAxSize := b.Size(bOtherAx)

	if size <= 1 || aOtherAxSize <= 1 || bOtherAxSize <= 1 {
		return TryTensorDot([]*nune.Tensor[T]{a, b}, []int{aAxis, bAxis})
	}

	as := make([]*nune.Tensor[T], 4)

	as[0] = View(a, makeAxisPairs(a.Shape(), map[int][2]int{aOtherAx: {0, aOtherAxSize / 2}, aAxis: {0, size / 2}}), nil)
	as[1] = View(a, makeAxisPairs(a.Shape(), map[int][2]int{aOtherAx: {0, aOtherAxSize / 2}, aAxis: {size / 2, size}}), nil)
	as[2] = View(a, makeAxisPairs(a.Shape(), map[int][2]int{aOtherAx: {aOtherAxSize / 2, aOtherAxSize}, aAxis: {0, size / 2}}), nil)
	as[3] = View(a, makeAxisPairs(a.Shape(), map[int][2]int{aOtherAx: {aOtherAxSize / 2, aOtherAxSize}, aAxis: {size / 2, size}}), nil)
	bs := make([]*nune.Tensor[T], 4)

	bs[0] = View(b, makeAxisPairs(b.Shape(), map[int][2]int{bOtherAx: {0, bOtherAxSize / 2}, bAxis: {0, size / 2}}), nil)
	bs[1] = View(b, makeAxisPairs(b.Shape(), map[int][2]int{bOtherAx: {bOtherAxSize / 2, bOtherAxSize}, bAxis: {0, size / 2}}), nil)
	bs[2] = View(b, makeAxisPairs(b.Shape(), map[int][2]int{bOtherAx: {0, bOtherAxSize / 2}, bAxis: {size / 2, size}}), nil)
	bs[3] = View(b, makeAxisPairs(b.Shape(), map[int][2]int{bOtherAx: {bOtherAxSize / 2, bOtherAxSize}, bAxis: {size / 2, size}}), nil)

	ts := make([]*nune.Tensor[T], 7)
	ss := make([]*nune.Tensor[T], 7)

	var wg sync.WaitGroup

	wg.Add(2)
	go func(as []*nune.Tensor[T], ts []*nune.Tensor[T]) {
		ts[0] = Sub(as[1], as[3])
		ts[1] = Add(as[0], as[3])
		ts[2] = Sub(as[0], as[2])
		ts[3] = Add(as[0], as[1])
		ts[4] = as[0]
		ts[5] = as[3]
		ts[6] = Add(as[2], as[3])
		wg.Done()
	}(as, ts)
	go func(bs []*nune.Tensor[T], ss []*nune.Tensor[T]) {
		ss[0] = Add(bs[2], bs[3])
		ss[1] = Add(bs[0], bs[3])
		ss[2] = Add(bs[0], bs[1])
		ss[3] = bs[3]
		ss[4] = Sub(bs[1], bs[3])
		ss[5] = Sub(bs[2], bs[0])
		ss[6] = bs[0]
		wg.Done()
	}(bs, ss)

	wg.Wait()

	wg.Add(7)
	for i := 0; i < 7; i++ {
		go func(t **nune.Tensor[T], s *nune.Tensor[T]) {
			*t = TensorDot([]*nune.Tensor[T]{*t, s}, []int{aAxis, bAxis})
			wg.Done()
		}(&ts[i], ss[i])
	}
	wg.Wait()

	var q01, q24 *nune.Tensor[T]
	wg.Add(2)
	go func(q2, q3, q4 *nune.Tensor[T], a1, q24 **nune.Tensor[T]) {
		*a1 = Add(q3, q4)
		*q24 = Sub(q4, q2)
		wg.Done()
	}(ts[2], ts[3], ts[4], &as[1], &q24)
	go func(q0, q1, q5, q6 *nune.Tensor[T], a2, q01 **nune.Tensor[T]) {
		*a2 = Add(q5, q6)
		*q01 = Add(q0, q1)
		wg.Done()
	}(ts[0], ts[1], ts[5], ts[6], &as[2], &q01)
	wg.Wait()

	wg.Add(2)
	go func(q01, q3, q5 *nune.Tensor[T], a0 **nune.Tensor[T]) {
		*a0 = Sub(q01, q3)
		AddAssign(*a0, q5)
		wg.Done()
	}(q01, ts[3], ts[5], &as[0])
	go func(q24, q1, q6 *nune.Tensor[T], a3 **nune.Tensor[T]) {
		*a3 = Add(q24, q1)
		SubAssign(*a3, q6)
		wg.Done()
	}(q24, ts[1], ts[6], &as[3])
	wg.Wait()

	shape := slices.Clone(as[0].Shape())
	newAxis := 0
	newAOtherAx, newBOtherAx := 0, 0
	for axis := 0; axis < a.Rank()+b.Rank(); axis++ {
		if axis == aAxis || axis == a.Rank()+bAxis {
			continue
		}
		if axis == aOtherAx {
			newAOtherAx = newAxis
		}
		if axis == a.Rank()+bOtherAx {
			newBOtherAx = newAxis
		}
		newAxis++
	}
	shape[newAOtherAx] = aOtherAxSize
	shape[newBOtherAx] = bOtherAxSize
	out := nune.Zeros[T](shape...)

	ViewAssign(&out, as[0], map[int][2]int{newAOtherAx: {0, aOtherAxSize / 2}, newBOtherAx: {0, bOtherAxSize / 2}}, nil)
	ViewAssign(&out, as[1], map[int][2]int{newAOtherAx: {0, aOtherAxSize / 2}, newBOtherAx: {bOtherAxSize / 2, bOtherAxSize}}, nil)
	ViewAssign(&out, as[2], map[int][2]int{newAOtherAx: {aOtherAxSize / 2, aOtherAxSize}, newBOtherAx: {0, bOtherAxSize / 2}}, nil)
	ViewAssign(&out, as[3], map[int][2]int{newAOtherAx: {aOtherAxSize / 2, aOtherAxSize}, newBOtherAx: {bOtherAxSize / 2, bOtherAxSize}}, nil)
	return &out, nil
}

// StrassenDot computes matrix multiplation using Strassen algorithm on given axes.
func StrassenDot[T Number](a *nune.Tensor[T], b *nune.Tensor[T], aAxis int, bAxis int) *nune.Tensor[T] {
	out, err := TryStrassenDot(a, b, aAxis, bAxis)
	if err != nil {
		panic(err)
	}
	return out
}

func makeAxisPairs(shape []int, addAxisPairs map[int][2]int) map[int][2]int {
	axisPairs := make(map[int][2]int)
	for axis, x := range shape {
		axisPairs[axis] = [2]int{0, x}
	}
	if addAxisPairs == nil {
		return axisPairs
	}
	for axis, pair := range addAxisPairs {
		axisPairs[axis] = pair
	}
	return axisPairs
}
