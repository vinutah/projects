#! /usr/bin/perl
use strict;
use warnings;
use POSIX;
#chdir 'data/training';
my $sampling_rate = $ARGV[0];
my $training_file = $ARGV[1];



my $filename_in = "$training_file";
my $filename_out = "$training_file"."_s";
open(FILE_IN, $filename_in) or die "cannot open $filename_in";
open(FILE_OUT, ">$filename_out"); 

my @line_array;
my @index_array;
my $line_count = `wc -l < $filename_in`;
my $line_count_new = ($line_count)*($sampling_rate);
while (my $line = <FILE_IN>) { chomp($line); push @line_array, $line;}
foreach my $i (1..floor($line_count_new)) { 
    my $number = int(rand($line_count));  
    push @index_array, $number;
}

foreach my $index (@index_array) { print FILE_OUT "$line_array[$index]\n"; }
close(FILE_OUT);
close(FILE_IN);
