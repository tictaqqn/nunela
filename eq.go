package nunela

import (
	"github.com/vorduin/nune"
	"github.com/vorduin/slices"
)

func Equal[T Number](x *nune.Tensor[T], y *nune.Tensor[T]) bool {
	if !slices.Equal(x.Shape(), y.Shape()) {
		return false
	}
	if !slices.Equal(x.Ravel(), y.Ravel()) {
		return false
	}
	return true
}
