package nunela_test

import (
	"testing"

	"github.com/bababax11/nunela"
	"github.com/vorduin/nune"
)

func TestView(t *testing.T) {
	cases := map[string]struct {
		in        nune.Tensor[int]
		axisPairs [][2]int
		expected  nune.Tensor[int]
	}{
		"simple 2D tensor": {
			nune.FromBuffer([]int{0, 1, 2, 3, 4, 5}).Reshape(3, 2),
			[][2]int{{1, 1}},
			nune.FromBuffer([]int{1, 3, 5}),
		},
		"3D tensor 1": {
			nune.Range[int](1, 121, 1).Reshape(10, 3, 4),
			[][2]int{{0, 3}},
			nune.Range[int](37, 49, 1).Reshape(3, 4),
		},
		"3D tensor 2": {
			nune.Range[int](1, 121, 1).Reshape(10, 3, 4),
			[][2]int{{1, 1}},
			nune.FromBuffer([]int{
				5, 6, 7, 8,
				17, 18, 19, 20,
				29, 30, 31, 32,
				41, 42, 43, 44,
				53, 54, 55, 56,
				65, 66, 67, 68,
				77, 78, 79, 80,
				89, 90, 91, 92,
				101, 102, 103, 104,
				113, 114, 115, 116,
			}).Reshape(10, 4),
		},

		"3D tensor 3": {
			nune.Range[int](1, 121, 1).Reshape(10, 3, 4),
			[][2]int{{2, 2}},
			nune.FromBuffer([]int{
				3, 7, 11,
				15, 19, 23,
				27, 31, 35,
				39, 43, 47,
				51, 55, 59,
				63, 67, 71,
				75, 79, 83,
				87, 91, 95,
				99, 103, 107,
				111, 115, 119,
			}).Reshape(10, 3),
		},
		"3D tensor with multiple axes": {
			nune.Range[int](1, 121, 1).Reshape(10, 3, 4),
			[][2]int{{0, 3}, {2, 2}},
			nune.FromBuffer([]int{
				39, 43, 47,
			}),
		},
	}
	for name, tt := range cases {
		t.Run(name, func(t *testing.T) {
			val := nunela.View(&tt.in, tt.axisPairs)
			if !nunela.Equal(val, &tt.expected) {
				t.Error(name, val.Ravel(), tt.expected.Ravel())
			}
		})
	}
}
