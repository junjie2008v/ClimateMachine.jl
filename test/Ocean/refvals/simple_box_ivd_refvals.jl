refVals = []
refPrecs = []

#! format: off
# SC ========== Test number 1 reference values and precision match template. =======
# SC ========== /home/jmc/cliMa/cliMa_update/test/Ocean/SplitExplicit/simple_box_ivd.jl test reference values ======================================
# BEGIN SCPRINT
# varr - reference values (from reference run)
# parr - digits match precision (hand edit as needed)
#
# [
#  [ MPIStateArray Name, Field Name, Maximum, Minimum, Mean, Standard Deviation ],
#  [         :                :          :        :      :          :           ],
# ]
varr = [
 [  "oce Q_3D",   "u[1]", -1.50907071392284014566e-01,  1.73781231809542469069e-01,  4.68096331745367338212e-03,  1.95920021000241043052e-02 ],
 [  "oce Q_3D",   "u[2]", -1.41693937396426050679e-01,  1.46243917821118285527e-01, -2.51530766341314954843e-03,  1.75572837799599749953e-02 ],
 [  "oce Q_3D",       :η, -7.42455782775250150429e-01,  3.69730515308098417471e-01, -2.12053967392545028277e-04,  2.72181302813857772804e-01 ],
 [  "oce Q_3D",       :θ,  5.28450920905325794569e-03,  9.92875001597955630928e+00,  2.49640763974860568908e+00,  2.18014403700940651021e+00 ],
 [   "oce aux",       :w, -1.93641508217356502321e-04,  1.64389722003552838422e-04,  4.30819768270874037293e-07,  1.44751504460569400376e-05 ],
 [   "oce aux",    :pkin, -8.85137385724656411412e+00,  0.00000000000000000000e+00, -3.26232394846976436753e+00,  2.50203581939964081471e+00 ],
 [   "oce aux",     :wz0, -2.13126994266848044583e-05,  3.03761966887832467763e-05, -1.29724204981140911883e-10,  7.79125622163024900916e-06 ],
 [   "oce aux", "u_d[1]", -1.47643597045305691173e-01,  1.16172928281001314188e-01, -3.40056303283133183873e-05,  1.18325769360269061892e-02 ],
 [   "oce aux", "u_d[2]", -1.41299443151196246760e-01,  1.40392329116052066995e-01, -8.68704744435173440418e-06,  1.16473030652601838852e-02 ],
 [   "oce aux", "ΔGu[1]", -1.82438482660877217949e-06,  1.76131450220044681262e-06, -3.58493343107863405131e-08,  2.63467783463959988175e-07 ],
 [   "oce aux", "ΔGu[2]", -1.21868069222541804602e-06,  2.22130916543395098495e-06,  1.31060034108175441821e-06,  6.57246261117217741340e-07 ],
 [   "oce aux",       :y,  0.00000000000000000000e+00,  4.00000000000000046566e+06,  2.00000000000000000000e+06,  1.15573163901915703900e+06 ],
 [ "baro Q_2D",   "U[1]", -1.28638134009455562534e+01,  6.10305774663233222554e+01,  4.70737872487603414839e+00,  1.49617848161145641228e+01 ],
 [ "baro Q_2D",   "U[2]", -2.82511172627316504702e+01,  6.47636399564029829889e+01, -2.47937561262140748752e+00,  1.34334006549456468349e+01 ],
 [ "baro Q_2D",       :η, -7.42661610672267546995e-01,  3.69774652167159434413e-01, -2.12086144227697549498e-04,  2.72331664923152527713e-01 ],
 [  "baro aux",  "Gᵁ[1]", -1.76131450220044674486e-03,  1.82438482660877217441e-03,  3.58493343107863313016e-05,  2.63480826100707878241e-04 ],
 [  "baro aux",  "Gᵁ[2]", -2.22130916543395113064e-03,  1.21868069222541802570e-03, -1.31060034108175450969e-03,  6.57278797255507452567e-04 ],
 [  "baro aux", "U_c[1]", -1.28418434709131652482e+01,  6.09600328717265398382e+01,  4.71373811044994983632e+00,  1.49477163957520815529e+01 ],
 [  "baro aux", "U_c[2]", -2.82747850743473101431e+01,  6.47179561285744426868e+01, -2.50652647306075504474e+00,  1.34314270881023727355e+01 ],
 [  "baro aux",     :η_c, -7.42455782775250150429e-01,  3.69730515308098417471e-01, -2.12053967392540691468e-04,  2.72194776802272997429e-01 ],
 [  "baro aux", "U_s[1]", -1.28420859836829617251e+01,  6.09610745104530664662e+01,  4.71375870095391835690e+00,  1.49479160464055684798e+01 ],
 [  "baro aux", "U_s[2]", -2.82756417530404320360e+01,  6.47178077635098105702e+01, -2.50659230751473804943e+00,  1.34314567895758454341e+01 ],
 [  "baro aux",     :η_s, -7.42463591584554216674e-01,  3.69734792872739193026e-01, -2.12048722047518373819e-04,  2.72195582254090906460e-01 ],
 [  "baro aux",  "Δu[1]", -4.52576313913315424631e-04,  4.01397574102695402134e-04, -1.17811088890548056881e-05,  1.01680622362620656394e-04 ],
 [  "baro aux",  "Δu[2]", -2.43950808154872009637e-04,  3.86395073848479621119e-04,  5.11424556860824746545e-05,  7.52605326838232742588e-05 ],
 [  "baro aux",  :η_diag, -7.39883203465319660985e-01,  3.68215473126410952620e-01, -2.11645242010587240206e-04,  2.71980481766462722781e-01 ],
 [  "baro aux",      :Δη, -3.57633535239532118766e-03,  4.28419930737145016053e-03, -4.08725381931473987174e-07,  1.12670824245143363224e-03 ],
 [  "baro aux",       :y,  0.00000000000000000000e+00,  4.00000000000000046566e+06,  2.00000000000000000000e+06,  1.15578885204060329124e+06 ],
]
parr = [
 [  "oce Q_3D",   "u[1]",    12,    12,    12,    12 ],
 [  "oce Q_3D",   "u[2]",    12,    12,    12,    12 ],
 [  "oce Q_3D",       :η,    12,    12,     8,    12 ],
 [  "oce Q_3D",       :θ,    12,    12,    12,    12 ],
 [   "oce aux",       :w,    12,    12,     8,    12 ],
 [   "oce aux",    :pkin,    12,    12,    12,    12 ],
 [   "oce aux",     :wz0,    12,    12,     8,    12 ],
 [   "oce aux", "u_d[1]",    12,    12,    12,    12 ],
 [   "oce aux", "u_d[2]",    12,    12,    12,    12 ],
 [   "oce aux", "ΔGu[1]",    12,    12,    12,    12 ],
 [   "oce aux", "ΔGu[2]",    12,    12,    12,    12 ],
 [   "oce aux",       :y,    12,    12,    12,    12 ],
 [ "baro Q_2D",   "U[1]",    12,    12,    12,    12 ],
 [ "baro Q_2D",   "U[2]",    12,    12,    12,    12 ],
 [ "baro Q_2D",       :η,    12,    12,     8,    12 ],
 [  "baro aux",  "Gᵁ[1]",    12,    12,    12,    12 ],
 [  "baro aux",  "Gᵁ[2]",    12,    12,    12,    12 ],
 [  "baro aux", "U_c[1]",    12,    12,    12,    12 ],
 [  "baro aux", "U_c[2]",    12,    12,    12,    12 ],
 [  "baro aux",     :η_c,    12,    12,     8,    12 ],
 [  "baro aux", "U_s[1]",    12,    12,    12,    12 ],
 [  "baro aux", "U_s[2]",    12,    12,    12,    12 ],
 [  "baro aux",     :η_s,    12,    12,     8,    12 ],
 [  "baro aux",  "Δu[1]",    12,    12,    12,    12 ],
 [  "baro aux",  "Δu[2]",    12,    12,    12,    12 ],
 [  "baro aux",  :η_diag,    12,    12,     8,    12 ],
 [  "baro aux",      :Δη,    12,    12,     8,    12 ],
 [  "baro aux",       :y,    12,    12,    12,    12 ],
]
# END SCPRINT
# SC ====================================================================================

    append!(refVals ,[ varr ] )
    append!(refPrecs,[ parr ] )

#! format: on
