# [DiagnosticsMachine](@id DiagnosticsMachine)

```@meta
CurrentModule = ClimateMachine.DiagnosticsMachine
```

```@docs
DiagnosticsMachine
DiagnosticsMachine.init
```

### Diagnostic variables

```@docs
DiagnosticsMachine.DiagnosticVar
```

### Kinds of diagnostic variables

```@docs
DiagnosticsMachine.PointwiseDiagnostic
DiagnosticsMachine.HorizontalAverage
DiagnosticsMachine.ScalarDiagnostic
```

### Creating diagnostic variables

```@docs
DiagnosticsMachine.@pointwise_diagnostic
DiagnosticsMachine.@horizontal_average
DiagnosticsMachine.@scalar_diagnostic
```

These use:

```@docs
DiagnosticsMachine.generate_dv_interface
DiagnosticsMachine.generate_dv_function
```

To generate:

```@docs
DiagnosticsMachine.dv_name
DiagnosticsMachine.dv_attrib
DiagnosticsMachine.dv_args
```

Diagnostic variable implementations must use the following type in
their arguments:

```@docs
DiagnosticsMachine.States
```

### Creating diagnostics groups

```@docs
DiagnosticsMachine.@diagnostics_group
```

### Defining new diagnostic variable kinds

New diagnostic variable kinds are being added. Currently, these
must define the following functions (this list and the semantics
of these functions are subject to change).

```@docs
DiagnosticsMachine.dv_dg_points_length
DiagnosticsMachine.dv_dg_points_index
DiagnosticsMachine.dv_dg_elems_length
DiagnosticsMachine.dv_dg_elems_index
DiagnosticsMachine.dv_dg_dimnames
DiagnosticsMachine.dv_dg_dimranges
DiagnosticsMachine.dv_op
DiagnosticsMachine.dv_reduce
```
