package nunela_test

import (
	"testing"

	"github.com/bababax11/nunela"
	"github.com/vorduin/nune"
)

func TestAt(t *testing.T) {
	cases := map[string]struct {
		in       nune.Tensor[int]
		at       []int
		expected int
	}{
		"simple 2D tensor": {
			nune.FromBuffer([]int{0, 1, 2, 3, 4, 5}).Reshape(3, 2),
			[]int{1, 1},
			3,
		},
		"3D tensor": {
			nune.Range[int](1, 121, 1).Reshape(10, 3, 4),
			[]int{3, 1, 2},
			43,
		},
	}
	for name, tt := range cases {
		t.Run(name, func(t *testing.T) {
			val := nunela.At(&tt.in, tt.at...)
			if val != tt.expected {
				t.Error(name, val, tt.expected)
			}
		})
	}
}
