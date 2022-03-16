package nunela

import (
	"fmt"
	"regexp"
	"strings"
)

func parseEinSum(equation string, tensorsCount int) ([][]int, error) {
	out := make([][]int, 0, tensorsCount)
	re := regexp.MustCompile(fmt.Sprintf("[a-zA-Z\\.]+,{%d}", tensorsCount))
	if !re.MatchString(equation) {
		return nil, NewErrInappropriateEinString(equation)
	}
	indicesArr := strings.Split(equation, ",")
	for _, indices := range indicesArr {
		var o []int
		for _, r := range indices {
			var x int
			if r == '.' {
				x = 0
			} else {
				x = int(r) - 'A' + 1
			}
			o = append(o, x)
		}
		out = append(out, o)
	}
	return out, nil
}
