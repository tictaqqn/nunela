package nunela

// Number is the set of all numeric types and their supersets.
type Number interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~float32 | ~float64
}

type Tensor[T Number] interface {
	// Iter returns an Iterator over the Tensor's data,
	// Iter() nune.Iterator[T]

	// Ravel returns the Tensor's view in its data buffer.
	Ravel() []T

	// Scalar returns the scalar equivalent of a rank 0 Tensor.
	// Panics if the Tensor's rank is not 0.
	Scalar() T

	// Numel returns the number of elements in the Tensor's data buffer.
	Numel() int

	// Rank returns the Tensor's rank
	// (the number of axes in the Tensor's shape).
	Rank() int

	// Shape returns a copy of the Tensor's shape.
	Shape() []int

	// Stride() []int

	// Size returns the Tensor's number of dimensions at
	// the given axis.
	// Panics if axis is out of (0, rank) bounds.
	Size(int) int
}
