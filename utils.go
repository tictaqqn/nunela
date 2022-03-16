package nunela

import (
	"math"
	"runtime"

	"github.com/vorduin/nune"
)

// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// configCPU returns the number of CPU cores to use
// depending on the data's size.
func configCPU(size int) int {
	if nune.EnvConfig.NumCPU != 0 {
		return int(math.Min(float64(size), float64(nune.EnvConfig.NumCPU)))
	}

	bias := float64(size) / 4096 // this is handcoded, therefore beautiful. or ugly

	if bias < 1 {
		return 1
	} else if bias < float64(runtime.NumCPU()) {
		return int(math.Round(bias))
	} else {
		return runtime.NumCPU()
	}
}
