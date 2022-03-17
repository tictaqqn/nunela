package nunela

import (
	"sync"

	"github.com/vorduin/nune"
)

func TryTensorDotWithOneAxis[T Number](tensors []*nune.Tensor[T], axes []int) (*nune.Tensor[T], error) {
	if len(tensors) != len(axes) {
		return nil, NewErrDifferentLen(tensors, axes)
	}
	if len(tensors) <= 1 {
		return nil, NewErrNotEnoughTensorsGiven()
	}
	lenAxis := tensors[0].Size(axes[0])
	for !every(tensors[1:], func(i int, t *nune.Tensor[T]) bool { return tensors[i].Size(axes[i]) == lenAxis }) {
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

func TensorDotWithOneAxis[T Number](tensors []*nune.Tensor[T], axes []int) *nune.Tensor[T] {
	out, err := TryTensorDotWithOneAxis(tensors, axes)
	if err != nil {
		panic(err)
	}
	return out
}

// TryStrassenDotAx computes matrix multiplation using Strassen algorithm with given axes.
func TryStrassenDotAx[T Number](a *nune.Tensor[T], b *nune.Tensor[T], aAxis int, bAxis int) (*nune.Tensor[T], error) {
	if a.Rank() <= 1 || b.Rank() <= 1 {
		return TryTensorDotWithOneAxis([]*nune.Tensor[T]{a, b}, []int{aAxis, bAxis})
	}
	aSize := a.Size(aAxis)
	bSize := b.Size(bAxis)
	if aSize != bSize {
		return nil, NewErrDifferentShapes(a, b)
	}
	aOtherAx := aAxis - 1
	if aAxis == -1 {
		aOtherAx = 1
	}
	bOtherAx := bAxis - 1
	if bAxis == -1 {
		bOtherAx = 1
	}
	aOtherAxSize := a.Size(aOtherAx)
	bOtherAxSize := b.Size(bOtherAx)

	if aSize <= 1 || bSize <= 1 || aOtherAxSize <= 1 || bOtherAxSize <= 1 {
		return TryTensorDotWithOneAxis([]*nune.Tensor[T]{a, b}, []int{aAxis, bAxis})
	}

	as := make([]*nune.Tensor[T], 4)

	as[0] = View(a, map[int][2]int{aOtherAx: {0, aOtherAxSize / 2}, aAxis: {0, aSize / 2}}, nil)
	as[1] = View(a, map[int][2]int{aOtherAx: {0, aOtherAxSize / 2}, aAxis: {aSize / 2, aSize}}, nil)
	as[2] = View(a, map[int][2]int{aOtherAx: {aOtherAxSize / 2, aOtherAxSize}, aAxis: {0, aSize / 2}}, nil)
	as[3] = View(a, map[int][2]int{aOtherAx: {aOtherAxSize / 2, aOtherAxSize}, aAxis: {aSize / 2, aSize}}, nil)

	bs := make([]*nune.Tensor[T], 4)

	bs[0] = View(a, map[int][2]int{bOtherAx: {0, bOtherAxSize / 2}, bAxis: {0, bSize / 2}}, nil)
	bs[1] = View(a, map[int][2]int{bOtherAx: {0, bOtherAxSize / 2}, bAxis: {bSize / 2, bSize}}, nil)
	bs[2] = View(a, map[int][2]int{bOtherAx: {bOtherAxSize / 2, bOtherAxSize}, bAxis: {0, bSize / 2}}, nil)
	bs[3] = View(a, map[int][2]int{bOtherAx: {bOtherAxSize / 2, bOtherAxSize}, bAxis: {bSize / 2, bSize}}, nil)

	ts := make([]*nune.Tensor[T], 7)
	ss := make([]*nune.Tensor[T], 7)

	var wg sync.WaitGroup

	wg.Add(2)
	go func(as []*nune.Tensor[T], ts []*nune.Tensor[T]) {
		ts[0] = Sub(as[1], as[3])
		ts[1] = Add(as[0], as[3])
		ts[2] = Add(as[0], as[2])
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
		go func(t, s *nune.Tensor[T]) {
			*t = *TensorDotWithOneAxis([]*nune.Tensor[T]{t, s}, []int{aAxis, bAxis})
			wg.Done()
		}(ts[i], ss[i])
	}
	wg.Wait()

	wg.Add(2)
	go func(q3, q4, a1 *nune.Tensor[T]) {
		a1 = Add(q3, q4)
		wg.Done()
	}(ts[3], ts[4], as[1])
	go func(q5, q6, a2 *nune.Tensor[T]) {
		a2 = Add(q5, q6)
		wg.Done()
	}(ts[5], ts[6], as[2])
	wg.Wait()

	as[0] = Add(Sub(Add(ts[0], ts[1]), ts[3]), ts[6])
	as[3] = Sub(Add(Sub(ts[1], ts[2]), ts[4]), ts[6])
	shape := make([]int, as[0].Rank())
	shape[aOtherAx] = aOtherAxSize
	shape[bOtherAx] = bOtherAxSize
	out := nune.Zeros[T](shape...)
	ViewAssign(&out, as[0], map[int][2]int{aOtherAx: {0, aOtherAxSize / 2}, bOtherAx: {0, bOtherAxSize / 2}}, nil)
	ViewAssign(&out, as[1], map[int][2]int{aOtherAx: {0, aOtherAxSize / 2}, bOtherAx: {bOtherAxSize / 2, bOtherAxSize}}, nil)
	ViewAssign(&out, as[2], map[int][2]int{aOtherAx: {aOtherAxSize / 2, aOtherAxSize}, bOtherAx: {0, bOtherAxSize / 2}}, nil)
	ViewAssign(&out, as[3], map[int][2]int{aOtherAx: {aOtherAxSize / 2, aOtherAxSize}, bOtherAx: {bOtherAxSize / 2, bOtherAxSize}}, nil)
	return &out, nil
}

func StrassenDotAx[T Number](a *nune.Tensor[T], b *nune.Tensor[T], aAxis int, bAxis int) *nune.Tensor[T] {
	out, err := TryStrassenDotAx(a, b, aAxis, bAxis)
	if err != nil {
		panic(err)
	}
	return out
}
