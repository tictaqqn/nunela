package nunela_test

import (
	"testing"

	"github.com/bababax11/nunela"
	"github.com/vorduin/nune"
)

func TestTensorDotWithOneAxis(t *testing.T) {
	nune.EnvConfig.Interactive = true
	cases := map[string]struct {
		tensor0  nune.Tensor[int]
		tensor1  nune.Tensor[int]
		axes     []int
		expected nune.Tensor[int]
	}{
		"matrix multiplation": {
			nune.Range[int](0, 8, 1).Reshape(2, 4),
			nune.Range[int](0, 12, 1).Reshape(4, 3),
			[]int{1, 0},
			nune.FromBuffer([]int{
				42, 48, 54,
				114, 136, 158,
			}).Reshape(2, 3),
		},
		"3D tensordot with different axes": {
			nune.Range[int](0, 8, 1).Reshape(2, 4),
			nune.Range[int](0, 12, 1).Reshape(2, 2, 3),
			[]int{0, 1},
			nune.FromBuffer([]int{
				12, 16, 20,
				36, 40, 44,

				15, 21, 27,
				51, 57, 63,

				18, 26, 34,
				66, 74, 82,

				21, 31, 41,
				81, 91, 101,
			}).Reshape(4, 2, 3),
		},
	}

	for name, tt := range cases {
		t.Run(name, func(t *testing.T) {
			val := nunela.TensorDot([]*nune.Tensor[int]{
				&tt.tensor0, &tt.tensor1,
			}, tt.axes)
			if !nunela.Equal(val, &tt.expected) {
				t.Error(name, val.Ravel(), tt.expected.Ravel())
			}
		})
	}
}

func TestStrassenDotAx(t *testing.T) {
	nune.EnvConfig.Interactive = true
	cases := map[string]struct {
		tensor0  nune.Tensor[int]
		tensor1  nune.Tensor[int]
		axes     []int
		expected nune.Tensor[int]
	}{
		"matrix multiplation": {
			nune.Range[int](0, 8, 1).Reshape(2, 4),
			nune.Range[int](0, 16, 1).Reshape(4, 4),
			[]int{1, 0},
			nune.FromBuffer([]int{
				56, 62, 68, 74,
				152, 174, 196, 218,
			}).Reshape(2, 4),
		},
		"3D tensordot with different axes": {
			nune.Range[int](0, 8, 1).Reshape(2, 4),
			nune.Range[int](0, 16, 1).Reshape(2, 2, 4),
			[]int{0, 1},
			nune.FromBuffer([]int{
				16, 20, 24, 28,
				48, 52, 56, 60,

				20, 26, 32, 38,
				68, 74, 80, 86,

				24, 32, 40, 48,
				88, 96, 104, 112,

				28, 38, 48, 58,
				108, 118, 128, 138,
			}).Reshape(4, 2, 4),
		},
		"matrix multiplation with odd sizes": {
			nune.Range[int](0, 8, 1).Reshape(2, 4),
			nune.Range[int](0, 12, 1).Reshape(4, 3),
			[]int{1, 0},
			nune.FromBuffer([]int{
				42, 48, 54,
				114, 136, 158,
			}).Reshape(2, 3),
		},
		"3D tensordot with different axes with odd sizes": {
			nune.Range[int](0, 8, 1).Reshape(2, 4),
			nune.Range[int](0, 12, 1).Reshape(2, 2, 3),
			[]int{0, 1},
			nune.FromBuffer([]int{
				12, 16, 20,
				36, 40, 44,

				15, 21, 27,
				51, 57, 63,

				18, 26, 34,
				66, 74, 82,

				21, 31, 41,
				81, 91, 101,
			}).Reshape(4, 2, 3),
		},
	}

	for name, tt := range cases {
		t.Run(name, func(t *testing.T) {
			val := nunela.StrassenDot(
				&tt.tensor0, &tt.tensor1, tt.axes[0], tt.axes[1])
			if !nunela.Equal(val, &tt.expected) {
				t.Error(name, val.Ravel(), tt.expected.Ravel())
			}
		})
	}
}

func BenchmarkTensorDot(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		nunela.TensorDot([]*nune.Tensor[TestsT]{&tensor, &tensor}, []int{1, 0})
	})
}

func BenchmarkStrassen(b *testing.B) {
	benchmarkOp(b, func(tensor nune.Tensor[TestsT]) {
		nunela.StrassenDot(&tensor, &tensor, 1, 0)
	})
}

func FuzzTensorDot(f *testing.F) {
	f.Add(2, 1, 3, 1, 4, 1, 5, 1)
	f.Fuzz(func(t *testing.T, x0, x1, x2, x3, y0, y1, y2, y3 int) {
		tensor0 := nune.FromBuffer([]int{x0, x1, x2, x3}).Reshape(2, 2)
		tensor1 := nune.FromBuffer([]int{y0, y1, y2, y3}).Reshape(2, 2)
		tensorDot := nunela.TensorDot([]*nune.Tensor[int]{&tensor0, &tensor1}, []int{1, 0})
		strassenDot := nunela.StrassenDot(&tensor0, &tensor1, 1, 0)
		if !nunela.Equal(tensorDot, strassenDot) {
			f.Failed()
		}
	})
}

func FuzzTensorDotWithDifferentShapes(f *testing.F) {
	f.Add(2, 1, 3, 1, 4, 1, 5, 1, 3)
	f.Fuzz(func(t *testing.T, start0, step0, shape0X, shape01Y, shape0Z, start1, step1, shape1X, shape1Z int) {
		if shape0X <= 0 || shape01Y <= 0 || shape0Z <= 0 || shape1X <= 0 || shape1Z <= 0 || step0 <= 0 || step1 <= 0 {
			return
		}
		tensor0 := nune.Range[int](start0, start0+shape0X*shape01Y*shape0Z*step0, step0).Reshape(shape0X, shape01Y, shape0Z)
		tensor1 := nune.Range[int](start1, start1+shape1X*shape01Y*shape1Z*step1, step1).Reshape(shape1X, shape01Y, shape1Z)
		tensorDot := nunela.TensorDot([]*nune.Tensor[int]{&tensor0, &tensor1}, []int{1, 1})
		strassenDot := nunela.StrassenDot(&tensor0, &tensor1, 1, 1)
		if !nunela.Equal(tensorDot, strassenDot) {
			f.Failed()
		}
	})
}
