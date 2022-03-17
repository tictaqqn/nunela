package nunela_test

import (
	"testing"

	"github.com/bababax11/nunela"
	"github.com/vorduin/nune"
)

func TestZipMul(t *testing.T) {
	nune.EnvConfig.Interactive = true
	cases := map[string]struct {
		tensor0  nune.Tensor[int]
		tensor1  nune.Tensor[int]
		expected nune.Tensor[int]
	}{
		"matrix element-wise multiplation": {
			nune.Range[int](0, 8, 1).Reshape(2, 4),
			nune.Range[int](3, 11, 1).Reshape(2, 4),
			nune.FromBuffer([]int{
				0, 4, 10, 18,
				28, 40, 54, 70,
			}).Reshape(2, 4),
		},
		// "scalor product": {
		// 	nune.Range[int](3, 4, 1).Reshape(),
		// 	nune.Range[int](8, 9, 1).Reshape(),
		// 	nune.FromBuffer([]int{24}).Reshape(),
		// },
	}

	for name, tt := range cases {
		t.Run(name, func(t *testing.T) {
			val := nunela.Mul(
				&tt.tensor0, &tt.tensor1)
			if !nunela.Equal(val, &tt.expected) {
				t.Error(name, val.Ravel(), tt.expected.Ravel())
			}
		})
	}
}

func TestZipAssignMul(t *testing.T) {
	nune.EnvConfig.Interactive = true
	cases := map[string]struct {
		tensor0  nune.Tensor[int]
		tensor1  nune.Tensor[int]
		expected nune.Tensor[int]
	}{
		"matrix element-wise multiplation": {
			nune.Range[int](0, 8, 1).Reshape(2, 4),
			nune.Range[int](3, 11, 1).Reshape(2, 4),
			nune.FromBuffer([]int{
				0, 4, 10, 18,
				28, 40, 54, 70,
			}).Reshape(2, 4),
		},
		// "scalor product": {
		// 	nune.Range[int](3, 4, 1).Reshape(),
		// 	nune.Range[int](8, 9, 1).Reshape(),
		// 	nune.FromBuffer([]int{24}).Reshape(),
		// },
	}

	for name, tt := range cases {
		t.Run(name, func(t *testing.T) {
			nunela.MulAssign(
				&tt.tensor0, &tt.tensor1)
			if !nunela.Equal(&tt.tensor0, &tt.expected) {
				t.Error(name, tt.tensor0.Ravel(), tt.expected.Ravel())
			}
		})
	}
}
