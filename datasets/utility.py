def get_windows_with_padding(tissue_acceptor_intron, tissue_donor_intron, tissue_acceptor_exon, tissue_donor_exon, seq, overhang):
            """
            Split seq for tissue specific predictions
            Args:
            seq: seqeunce to split
            overhang: (intron_length acceptor side, intron_length donor side) of
                        the input sequence
            """

            (acceptor_intron, donor_intron) = overhang

            assert acceptor_intron <= len(seq), "Input sequence acceptor intron" \
                " length cannot be longer than the input sequence"
            assert donor_intron <= len(seq), "Input sequence donor intron length" \
                " cannot be longer than the input sequence"

            # need to pad N if seq not enough long
            diff_acceptor = acceptor_intron - tissue_acceptor_intron
            if diff_acceptor < 0:
                seq = "N" * abs(diff_acceptor) + seq
            elif diff_acceptor > 0:
                seq = seq[diff_acceptor:]

            diff_donor = donor_intron - tissue_donor_intron
            if diff_donor < 0:
                seq = seq + "N" * abs(diff_donor)
            elif diff_donor > 0:
                seq = seq[:-diff_donor]

            return {
                'acceptor': seq[:tissue_acceptor_intron
                                + tissue_acceptor_exon],
                'donor': seq[-tissue_donor_exon
                            - tissue_donor_intron:]
            }

def get_windows_with_padding_intronOnly(tissue_acceptor_intron, tissue_donor_intron, tissue_acceptor_exon, tissue_donor_exon, seq, overhang):
        """
        Split seq for tissue specific predictions
        Args:
        seq: seqeunce to split
        overhang: (intron_length acceptor side, intron_length donor side) of
                    the input sequence
        """

        (acceptor_intron, donor_intron) = overhang

        assert acceptor_intron <= len(seq), "Input sequence acceptor intron" \
            " length cannot be longer than the input sequence"
        assert donor_intron <= len(seq), "Input sequence donor intron length" \
            " cannot be longer than the input sequence"

        # need to pad N if seq not enough long
        diff_acceptor = acceptor_intron - tissue_acceptor_intron
        if diff_acceptor < 0:
            seq = "N" * abs(diff_acceptor) + seq
        elif diff_acceptor > 0:
            seq = seq[diff_acceptor:]

        diff_donor = donor_intron - tissue_donor_intron
        if diff_donor < 0:
            seq = seq + "N" * abs(diff_donor)
        elif diff_donor > 0:
            seq = seq[:-diff_donor]

        left_intron_len  = tissue_acceptor_intron
        right_intron_len = tissue_donor_intron

        # Slice intron-only windows (exclude exon completely)
        acceptor_window = seq[:left_intron_len]                      # last bases before exon
        donor_window    = seq[-right_intron_len:]                    # first bases after exon

        return {
            'acceptor': acceptor_window,
            'donor': donor_window,
        }