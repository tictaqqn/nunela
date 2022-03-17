// Copyright © The Nune Author. All rights reserved.

package nunela_test

import (
	"testing"
	"time"

	"github.com/vorduin/nune"
)

func benchmarkMicro(b *testing.B, f func()) {
	b.ResetTimer()

	start := time.Now()
	for i := 0; i < b.N; i++ {
		f()
	}
	execTime := time.Since(start)

	b.ReportMetric(0, "ns/op")
	b.ReportMetric((1e6*execTime.Seconds())/float64(b.N), "μs/op")
}

func benchmarkMilli(b *testing.B, f func()) {
	b.ResetTimer()

	start := time.Now()
	for i := 0; i < b.N; i++ {
		f()
	}
	execTime := time.Since(start)

	b.ReportMetric(0, "ns/op")
	b.ReportMetric((1e3*execTime.Seconds())/float64(b.N), "ms/op")
}

type TestsT float64

func newTensor(num int) nune.Tensor[TestsT] {
	return nune.Range[TestsT](0, num*num, 1).Reshape(num, num)
}

func benchmarkOp(b *testing.B, f func(nune.Tensor[TestsT])) {
	b.Run("1e4Float64Procs1", func(b *testing.B) {
		nune.EnvConfig.NumCPU = 1
		tensor := newTensor(1e2)

		benchmarkMilli(b, func() {
			f(tensor)
		})
	})

	b.Run("1e4Float64ProcsN", func(b *testing.B) {
		nune.EnvConfig.NumCPU = 0
		tensor := newTensor(1e2)

		benchmarkMilli(b, func() {
			f(tensor)
		})
	})

	b.Run("1.6e5Float64Procs1", func(b *testing.B) {
		nune.EnvConfig.NumCPU = 1
		tensor := newTensor(2e2)

		benchmarkMilli(b, func() {
			f(tensor)
		})
	})

	b.Run("1.6e5Float64ProcsN", func(b *testing.B) {
		nune.EnvConfig.NumCPU = 0
		tensor := newTensor(2e2)

		benchmarkMilli(b, func() {
			f(tensor)
		})
	})
}
