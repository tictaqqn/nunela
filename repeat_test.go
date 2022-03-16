package nunela_test

import (
	"testing"

	"github.com/bababax11/nunela"
	"github.com/vorduin/nune"
)

func TestRepeat(t *testing.T) {
	cases := map[string]struct {
		in       nune.Tensor[int]
		baseAxes []int
		toShape  []int
		expected nune.Tensor[int]
	}{
		"simple 2D tensor": {
			nune.FromBuffer([]int{0, 1, 2, 3, 4, 5}).Reshape(3, 2),
			[]int{1, 0},
			[]int{2, 3, 2},
			nune.FromBuffer([]int{0, 0, 2, 2, 4, 4, 1, 1, 3, 3, 5, 5}).Reshape(2, 3, 2),
		},
	}
	for name, tt := range cases {
		t.Run(name, func(t *testing.T) {
			val := nunela.Repeat(&tt.in, tt.baseAxes, tt.toShape)
			if !nunela.Equal(val, &tt.expected) {
				t.Error(name, val.Ravel(), tt.expected.Ravel())
			}
		})
	}
}
