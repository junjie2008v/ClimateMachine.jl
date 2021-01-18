##### Prognostic variable

import ..BalanceLaws: n_components
export Mass, Momentum, Energy
export Moisture, TotalMoisture, LiquidMoisture, IceMoisture
export Precipitation, Rain, Snow
export Tracers

struct Mass <: PrognosticVariable end
struct Momentum <: PrognosticVariable end
n_components(::Momentum) = 3

struct Energy <: PrognosticVariable end

abstract type Moisture <: PrognosticVariable end
struct TotalMoisture <: Moisture end
struct LiquidMoisture <: Moisture end
struct IceMoisture <: Moisture end

abstract type Precipitation <: PrognosticVariable end
struct Rain <: Precipitation end
struct Snow <: Precipitation end

struct Tracers{N} <: PrognosticVariable end
n_components(::Tracers{N}) where {N} = N
