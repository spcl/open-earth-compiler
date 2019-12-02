func @lap(%in : !sten.view<?x?x?xf64>) -> f64
  attributes { sten.function } {
	%0 = sten.access %in[-1, 0, 0] :  !sten.view<?x?x?xf64>
	return %0 : f64
}
